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
# 1. 설정 및 경로 (D드라이브 전용)
# ==========================================
BASE_DIR = Path(r"D:/아카이브.ver2").resolve()
REPORT_PATH = BASE_DIR / "evaluation_report_epoch7.csv" 
GPT_JSON_PATH = BASE_DIR / "caption.json"

# 체크포인트 경로 설정
MODEL_LOAD_PATH = BASE_DIR / "checkpoints/epoch7_final_checkpoints/final_step_35000"
MODEL_SAVE_PATH = BASE_DIR / "checkpoints" / "blip_finetuned_epoch8"
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "epoch8_checkpoints"
TRAIN_IMG_DIR = BASE_DIR / "train"

# [메모리 최적화] 배치 사이즈를 1로 고정하여 공유 메모리 사용 방지
LEARNING_RATE = 1e-6
BATCH_SIZE = 1 
EPOCHS = 1
SAVE_STEPS = 7000
TARGET_ACCURACY = 96.5

# ==========================================
# 2. GPT 데이터 로드 및 캡션 믹싱 함수
# ==========================================
with open(GPT_JSON_PATH, 'r', encoding='utf-8') as f:
    gpt_captions = json.load(f)

def get_epoch8_caption(cls_folder_name):
    # 폴더명에서 클래스명 추출
    clean_name = cls_folder_name.split('.')[-1] if '.' in cls_folder_name else cls_folder_name
    
    # [전략] 60% GPT 고급 문장 / 40% 기존 안정적 템플릿 (3종)
    if clean_name in gpt_captions and random.random() < 0.6:
        return random.choice(gpt_captions[clean_name])
    else:
        templates = [
            f"a photo of a {clean_name.replace('_', ' ')}",
            f"this image shows a {clean_name.replace('_', ' ')}",
            f"the {clean_name.replace('_', ' ')} is clearly visible in the scene"
        ]
        return random.choice(templates)

# ==========================================
# 3. 데이터셋 구성
# ==========================================
class Epoch8Dataset(Dataset):
    def __init__(self, img_dir, report_path, processor):
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.samples = []

        df = pd.read_csv(report_path)
        class_scores = df.groupby('class')['is_correct'].mean() * 100
        
        for class_dir in tqdm(self.img_dir.iterdir(), desc="데이터 리스트 생성"):
            if class_dir.is_dir():
                cls_folder_name = class_dir.name
                cls_display_name = cls_folder_name.replace("_", " ")

                score = class_scores.get(cls_display_name, 100)
                # 가중치 복습 (단위 축소로 메모리 부담 경감)
                if score < 90.0: repeat = 5
                elif score < TARGET_ACCURACY: repeat = 2
                else: repeat = 1

                img_paths = list(class_dir.glob("*"))
                for img_path in img_paths:
                    if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                        for _ in range(repeat):
                            self.samples.append((img_path, cls_folder_name))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_folder_name = self.samples[idx]
        caption = get_epoch8_caption(cls_folder_name)
        
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
                # max_length 제한으로 메모리 확보
                encoding = self.processor(images=image, text=caption, padding="max_length", truncation=True, max_length=50, return_tensors="pt")
            return {k: v.squeeze(0) for k, v in encoding.items()}
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# ==========================================
# 4. 학습 실행
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained(MODEL_LOAD_PATH)
model = BlipForConditionalGeneration.from_pretrained(MODEL_LOAD_PATH).to(device)

dataset = Epoch8Dataset(TRAIN_IMG_DIR, REPORT_PATH, processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

scaler = torch.cuda.amp.GradScaler()
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

model.train()
print(f"\n🚀 [메모리 최적화 모드] Epoch 8 학습 시작 (Batch Size: {BATCH_SIZE})")

for epoch in range(EPOCHS):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for i, batch in enumerate(pbar):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
            loss = outputs.loss
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if (i+1) % 100 == 0:
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'step': i+1})
            # [핵심] 100스텝마다 메모리 강제 청소
            torch.cuda.empty_cache()
            gc.collect()

        if (i+1) % SAVE_STEPS == 0:
            checkpoint_path = CHECKPOINT_DIR / f"epoch8_step_{i+1}"
            model.save_pretrained(checkpoint_path)
            processor.save_pretrained(checkpoint_path)

# 최종 저장
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
model.save_pretrained(MODEL_SAVE_PATH)
processor.save_pretrained(MODEL_SAVE_PATH)
print(f"\n✨ 학습 완료! D:/아카이브.ver2/checkpoints/blip_finetuned_epoch8_final 경로를 확인하세요.")