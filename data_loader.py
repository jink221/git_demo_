import os
import json
import torch
import pickle
import nltk
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path


# 1. 전역 설정 (Global Configuration)
CONFIG = {
    # 경로 설정
    "vocab_path": "vocab.pkl",
    "img_dir": "./data/images/train",
    "json_dir": "./data/captions/train.json",
    "nltk_data_dir": "nltk_data",
    
    # 하이퍼파라미터
    "batch_size": 4,
    "num_workers": 2,
    "image_size": (224, 224),
    "shuffle": True,
    
    # 이미지 정규화 값 (ImageNet 기준)
    "norm_mean": (0.485, 0.456, 0.406),
    "norm_std": (0.229, 0.224, 0.225)
}

# NLTK 데이터 경로 설정
base_dir = Path(__file__).resolve().parent
custom_nltk_path = str(base_dir / CONFIG["nltk_data_dir"])

if custom_nltk_path not in nltk.data.path:
    nltk.data.path.insert(0, custom_nltk_path)


# 2. 클래스 및 함수 정의
class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}

    def __len__(self):
        return len(self.itos)

    def numericalize(self, text):
        tokenized_text = nltk.word_tokenize(text.lower())
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]


class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, json_path, vocab, transform=None):
        self.root_dir = Path(root_dir)
        self.vocab = vocab
        self.transform = transform
        
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_id = item['image']
        caption = item['caption']
        
        img_path = self.root_dir / img_id
        image = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<START>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<END>"])

        return image, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        return images, targets


def get_loader(config):
    # 1. 단어 사전 로드
    with open(config["vocab_path"], "rb") as f:
        vocab = pickle.load(f)

    # 2. 이미지 변환 설정 (Config 값 참조)
    transform = transforms.Compose([
        transforms.Resize(config["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(config["norm_mean"], config["norm_std"])
    ])

    # 3. 데이터셋 인스턴스
    dataset = ImageCaptionDataset(
        root_dir=config["img_dir"], 
        json_path=config["json_dir"], 
        vocab=vocab, 
        transform=transform
    )

    # 4. 데이터로더 생성
    pad_idx = vocab.stoi["<PAD>"]
    loader = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=config["shuffle"],
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    return loader, dataset


if __name__ == "__main__":
    # 필요한 파일 존재 여부 확인
    v_exists = os.path.exists(CONFIG["vocab_path"])
    j_exists = os.path.exists(CONFIG["train_json"])

    if v_exists and j_exists:
        # CONFIG 딕셔너리만 넘겨주면 로더가 생성됩니다.
        train_loader, train_dataset = get_loader(CONFIG)

        print(f"로딩 완료! 배치 사이즈: {CONFIG['batch_size']}")

        # 샘플 배치 확인
        for imgs, caps in train_loader:
            print(f"이미지 배치 크기: {imgs.shape}") 
            print(f"캡션 배치 크기: {caps.shape}") 
            break
    else:
        print(f"[ERROR] 경로를 확인하세요. (VOCAB: {v_exists}, JSON: {j_exists})")