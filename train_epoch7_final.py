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
# 1. 경로 및 설정 (Epoch 10 결과물 기반)
# ==========================================
BASE_DIR = Path(r"D:/아카이브.ver2").resolve()
REPORT_PATH = BASE_DIR / "evaluation_report_epoch10.csv" 
GPT_JSON_PATH = BASE_DIR / "caption.json"
MODEL_LOAD_PATH = BASE_DIR / "checkpoints/blip_finetuned_epoch10" 
MODEL_SAVE_PATH = BASE_DIR / "checkpoints/blip_finetuned_epoch11" # 최종 저장 폴더
TRAIN_IMG_DIR = BASE_DIR / "train"

# 하이퍼파라미터 (밤샘 학습용 정밀 설정)
LEARNING_RATE = 2e-7        # 더 낮춰서 기존 지식 파괴 방지
BATCH_SIZE = 1 
EPOCHS = 2                  
SAVE_STEPS = 8000

# GPT 문장 로드
with open(GPT_JSON_PATH, 'r', encoding='utf-8') as f:
    gpt_captions = json.load(f)

# ==========================================
# 2. 정밀 보강 데이터셋 (약점 극복 + 문장력 심화)
# ==========================================
class NightDeepLearningDataset(Dataset):
    def __init__(self, img_dir, report_path, processor):
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.samples = []
        
        df = pd.read_csv(report_path)
        class_scores = df.groupby('class')['is_correct'].mean() * 100
        
        for class_dir in tqdm(self.img_dir.iterdir(), desc="취약 클래스 분석 및 로드"):
            if class_dir.is_dir():
                cls_folder_name = class_dir.name
                report_cls_name = cls_folder_name.replace("_", " ")
                score = class_scores.get(report_cls_name, 100)
                
                # [복습 강도 설정]
                if score < 85.0:
                    repeat = 3  # lobster 등 최하위는 3번씩 더 보기
                elif score < 95.0:
                    repeat = 2  # 중하위권은 2번씩 더 보기
                else:
                    repeat = 1
                
                img_paths = list(class_dir.glob("*"))
                for img_path in img_paths:
                    if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                        for _ in range(repeat):
                            self.samples.append((img_path, cls_folder_name))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_folder_name = self.samples[idx]
        
        # [9.5 : 0.5 전략] 문장 퀄리티를 위해 GPT 비중을 극대화
        if random.random() < 0.95:
            captions_list = gpt_captions.get(cls_folder_name, [])
            caption = random.choice(captions_list).replace("_", " ").strip()
        else:
            clean_name = cls_folder_name.replace("_", " ")
            caption = f"a professional clear photo of {clean_name}."

        image = Image.open(img_path).convert("RGB")
        encoding = self.processor(images=image, text=caption, padding="max_length", truncation=True, max_length=70, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in encoding.items()}

# ==========================================
# 3. 학습 루프 (Auto-Save 포함)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained(MODEL_LOAD_PATH)
model = BlipForConditionalGeneration.from_pretrained(MODEL_LOAD_PATH).to(device)

dataset = NightDeepLearningDataset(TRAIN_IMG_DIR, REPORT_PATH, processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

model.train()
print(f"🌙 밤샘 보강 학습 시작: 총 {EPOCHS} 에포크 진행 예정")

for epoch in range(EPOCHS):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for i, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'step': i+1})
        
        if (i+1) % SAVE_STEPS == 0:
            chk_path = BASE_DIR / "checkpoints" / f"night_train_epoch_{epoch+1}_step_{i+1}"
            model.save_pretrained(chk_path)

# 최종 저장
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
model.save_pretrained(MODEL_SAVE_PATH)
processor.save_pretrained(MODEL_SAVE_PATH)
print(f"✨ 모든 보강 학습이 끝났습니다! 모델 저장 위치: {MODEL_SAVE_PATH}")