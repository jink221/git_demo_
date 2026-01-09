import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import random
import json
import gc
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration

# ==========================================
# 1. 설정 및 경로 (수정된 파일명 반영)
# ==========================================
BASE_DIR = Path(r"D:/아카이브.ver2").resolve()
# 95%였던 Epoch 7 성적표를 기준으로 약점 보완
REPORT_PATH = BASE_DIR / "evaluation_report_epoch7.csv" 
GPT_JSON_PATH = BASE_DIR / "caption.json"

# [중요] 95% 성적을 냈던 최상의 상태인 epoch7 체크포인트를 로드합니다.
MODEL_LOAD_PATH = BASE_DIR / "checkpoints" / "epoch7_final_checkpoints" / "final_step_35000"
MODEL_SAVE_PATH = BASE_DIR / "checkpoints" / "blip_finetuned_epoch9_final"
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "epoch9_checkpoints"
TRAIN_IMG_DIR = BASE_DIR / "train"

# [학습 설정] 14시간을 충분히 활용하는 안전 세팅
LEARNING_RATE = 7e-7   # 기존보다 더 낮춰서 정확도 파괴 방지
BATCH_SIZE = 1         # GPU 메모리 안전 확보
EPOCHS = 2             # 14시간이면 전체 데이터를 2바퀴 꼼꼼히 학습 가능
SAVE_STEPS = 7000
# ==========================================
# 2. 캡션 생성 (정확도 80% : 고급어휘 20% 전략)
# ==========================================
with open(GPT_JSON_PATH, 'r', encoding='utf-8') as f:
    gpt_captions = json.load(f)

def get_epoch9_caption(cls_folder_name):
    clean_name = cls_folder_name.split('.')[-1] if '.' in cls_folder_name else cls_folder_name
    
    # [전략] 정확도를 지키기 위해 GPT 문장 비중을 20%로 낮춤 (환각 방지)
    if clean_name in gpt_captions and random.random() < 0.2:
        return random.choice(gpt_captions[clean_name])
    else:
        # 모델에게 사물 이름을 다시 확실히 각인시키는 기본 템플릿
        templates = [
            f"a photo of a {clean_name.replace('_', ' ')}",
            f"this image shows a {clean_name.replace('_', ' ')}",
            f"a clear shot of a {clean_name.replace('_', ' ')}"
        ]
        return random.choice(templates)

# ==========================================
# 3. 데이터셋 (Epoch 7의 약점 클래스 집중 공략)
# ==========================================
class Epoch9Dataset(Dataset):
    def __init__(self, img_dir, report_path, processor):
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.samples = []

        df = pd.read_csv(report_path)
        class_scores = df.groupby('class')['is_correct'].mean() * 100
        
        for class_dir in tqdm(self.img_dir.iterdir(), desc="복구 데이터셋 구성 중"):
            if class_dir.is_dir():
                cls_folder_name = class_dir.name
                cls_display_name = cls_folder_name.replace("_", " ")

                score = class_scores.get(cls_display_name, 100)
                # 95% 미만인 클래스는 5배 더 많이 보여주며 복습시킴
                repeat = 5 if score < 95.0 else 1

                img_paths = list(class_dir.glob("*"))
                for img_path in img_paths:
                    if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                        for _ in range(repeat):
                            self.samples.append((img_path, cls_folder_name))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_folder_name = self.samples[idx]
        caption = get_epoch9_caption(cls_folder_name)
        with Image.open(img_path) as img:
            image = img.convert("RGB")
            # 메모리 최적화를 위해 max_length 60 제한
            encoding = self.processor(images=image, text=caption, padding="max_length", truncation=True, max_length=60, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in encoding.items()}

# ==========================================
# 4. 학습 실행
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained(MODEL_LOAD_PATH)
model = BlipForConditionalGeneration.from_pretrained(MODEL_LOAD_PATH).to(device)

dataset = Epoch9Dataset(TRAIN_IMG_DIR, REPORT_PATH, processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
model.train()

print(f"\n🚀 [긴급복구] Epoch 7(final_step_35000) 기반으로 Epoch 9 학습을 시작합니다.")

for epoch in range(EPOCHS):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for i, batch in enumerate(pbar):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            loss = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids).loss
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if (i+1) % 100 == 0:
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'step': i+1})
            torch.cuda.empty_cache()
            gc.collect()

        if (i+1) % SAVE_STEPS == 0:
            checkpoint_path = CHECKPOINT_DIR / f"epoch9_step_{i+1}"
            model.save_pretrained(checkpoint_path)
            processor.save_pretrained(checkpoint_path)

MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
model.save_pretrained(MODEL_SAVE_PATH)
processor.save_pretrained(MODEL_SAVE_PATH)
print(f"\n✨ 복구 및 강화 학습 완료! 아침에 뵙겠습니다.")