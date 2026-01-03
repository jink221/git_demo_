import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from pathlib import Path

from data_loader import get_loader, Vocabulary
from model import CNNtoRNN

# 1. 전역 설정
CONFIG = {
    # 사용할 엔코더 선택: resnet50, efficientnet_b0, efficientnet_b1, vit_b_16, convnext_t, swin_t
    "encoder_type": "convnext_t", 
    
    "vocab_path": "vocab.pkl",
    "train_img_dir": "./data/images/train",
    "train_json": "./data/captions/train.json",
    "val_img_dir": "./data/images/val",
    "val_json": "./data/captions/val.json",
    "save_base_dir": "./checkpoints", 
    
    "optimizer_type": "AdamW",
    "learning_rate": 3e-4,
    "epochs": 30,
    "batch_size": 256,
    "num_workers": 8,
    "shuffle": True,
    "image_size": (224, 224),
    "norm_mean": (0.485, 0.456, 0.406),
    "norm_std": (0.229, 0.224, 0.225)
}

# 2. 엔코더 타입에 따른 최적 디코더 하이퍼파라미터 자동 설정 함수
def get_model_config(encoder_type):
    if "efficientnet" in encoder_type:
        return {"decoder_type": "gru", "embed_size": 256, "hidden_size": 512, "num_layers": 1}
    elif "vit" in encoder_type or "swin" in encoder_type:
        # Transformer 계열은 더 깊고 큰 디코더 권장
        return {"decoder_type": "lstm", "embed_size": 512, "hidden_size": 768, "num_layers": 2}
    else: # resnet, convnext 등
        return {"decoder_type": "lstm", "embed_size": 300, "hidden_size": 512, "num_layers": 1}

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 사용 디바이스: {device}")

    # 엔코더에 따른 세부 설정 로드
    m_cfg = get_model_config(CONFIG["encoder_type"])
    CONFIG.update(m_cfg)

    # 데이터 로더
    train_loader, train_dataset = get_loader({**CONFIG, "img_dir": CONFIG["train_img_dir"], "json_dir": CONFIG["train_json"]})
    val_loader, _ = get_loader({**CONFIG, "img_dir": CONFIG["val_img_dir"], "json_dir": CONFIG["val_json"], "shuffle": False})
    
    vocab = train_dataset.vocab
    vocab_size = len(vocab)

    # 모델 생성 (encoder_type과 decoder_type 명시적 전달)
    model = CNNtoRNN(
        embed_size=CONFIG["embed_size"], 
        hidden_size=CONFIG["hidden_size"], 
        vocab_size=vocab_size, 
        num_layers=CONFIG["num_layers"],
        encoder_type=CONFIG["encoder_type"],
        decoder_type=CONFIG["decoder_type"]
    ).to(device)

    run_name = f"{model.encoder_type}_{model.decoder_type}_ep{CONFIG['epochs']}_AdamW"
    model_filename = f"best_{model.encoder_type}_{model.decoder_type}_AdamW.pth"

    wandb.init(project="Image_Captioning", config=CONFIG, name=run_name)
    
    save_path = Path(CONFIG["save_base_dir"])
    save_path.mkdir(parents=True, exist_ok=True)
    best_model_path = save_path / model_filename

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.01)
    # resnet50, efficientnet_b0/b1, convnext_t -> 0.01
    # vit_b_16, swin_t -> 0.05

    best_val_loss = float('inf') 

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_train_loss = 0
        for imgs, captions in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            imgs, captions = imgs.to(device), captions.to(device)
            outputs = model(imgs, captions)
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        # Validation
        model.eval()
        total_val_loss, val_correct, val_tokens = 0, 0, 0
        with torch.no_grad():
            for imgs, captions in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                imgs, captions = imgs.to(device), captions.to(device)
                outputs = model(imgs, captions)
                loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
                total_val_loss += loss.item()

                _, predicted = outputs.max(2)
                mask = captions != vocab.stoi["<PAD>"]
                val_correct += (predicted[mask] == captions[mask]).sum().item()
                val_tokens += mask.sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = (val_correct / val_tokens) * 100

        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_acc": val_acc})
        print(f"Epoch {epoch} | Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 저장할 데이터
            save_data = {
                'model_state_dict': model.state_dict(),
                'encoder_type': model.encoder_type,
                'decoder_type': model.decoder_type,
                'embed_size': CONFIG['embed_size'],
                'hidden_size': CONFIG['hidden_size'],
                'num_layers': CONFIG['num_layers'],
                'vocab_size': vocab_size,
                'config': CONFIG
            }
            torch.save(save_data, best_model_path)
            print("  >> Best Model Saved!")

    wandb.finish()

if __name__ == "__main__":
    train()