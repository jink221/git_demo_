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
# 1. 경로 및 설정 (Epoch 4 기반 최적화)
# ==========================================
BASE_DIR = Path(r"D:/아카이브.ver2").resolve()
REPORT_PATH = BASE_DIR / "evaluation_report_epoch4.csv"  # 보유하신 epoch4 리포트
GPT_JSON_PATH = BASE_DIR / "caption.json"
MODEL_LOAD_PATH = BASE_DIR / "checkpoints" / "blip_finetuned_epoch4"
MODEL_SAVE_PATH = BASE_DIR / "checkpoints" / "blip_finetuned_epoch10"
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "epoch10_checkpoints"
TRAIN_IMG_DIR = BASE_DIR / "train"

# 하이퍼파라미터 (안정적인 이식을 위해 낮게 설정)
LEARNING_RATE = 5e-7 
BATCH_SIZE = 1 
EPOCHS = 1  # 문장 완성도를 위해 2에포크 권장 (14시간 충분)
SAVE_STEPS = 5000

# ==========================================
# 2. GPT 캡션 로드 및 전처리
# ==========================================
with open(GPT_JSON_PATH, 'r', encoding='utf-8') as f:
    gpt_captions = json.load(f)

def get_epoch10_caption(cls_folder_name):
    # GPT 문장 리스트 가져오기 (언더바 포함된 키 대응)
    captions_list = gpt_captions.get(cls_folder_name, [])
    
    # 90% 비율로 GPT 고급 문장 사용
    if captions_list and random.random() < 0.9:
        raw_caption = random.choice(captions_list)
        # 언더바(_) 제거하여 자연스러운 문장으로 변환
        return raw_caption.replace("_", " ").strip()
    else:
        # 10% 비율로 기본 이름 각인 (안전장치)
        clean_name = cls_folder_name.replace("_", " ")
        return f"a high-quality professional photo of a {clean_name}."

# ==========================================
# 3. 데이터셋 정의 (9:1 및 오답 가중치)
# ==========================================
class Epoch10Dataset(Dataset):
    def __init__(self, img_dir, report_path, processor):
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.samples = []
        
        # 리포트 로드 및 오답 클래스 분석
        df = pd.read_csv(report_path)
        # 클래스명 공백 제거 등 전처리 후 점수 계산
        df['class'] = df['class'].str.strip()
        class_scores = df.groupby('class')['is_correct'].mean() * 100
        
        for class_dir in tqdm(self.img_dir.iterdir(), desc="데이터셋 구성 중"):
            if class_dir.is_dir():
                cls_folder_name = class_dir.name
                # 리포트상의 클래스 이름 형식에 맞춤 (언더바 -> 공백)
                report_cls_name = cls_folder_name.replace("_", " ")
                score = class_scores.get(report_cls_name, 100)
                
                # 정확도 95% 미만 클래스는 3배 복습
                repeat = 3 if score < 95.0 else 1
                
                img_paths = list(class_dir.glob("*"))
                for img_path in img_paths:
                    if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                        for _ in range(repeat):
                            self.samples.append((img_path, cls_folder_name))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls_folder_name = self.samples[idx]
        caption = get_epoch10_caption(cls_folder_name)
        
        image = Image.open(img_path).convert("RGB")
        # GPT 문장 길이를 고려하여 max_length 60으로 확장
        encoding = self.processor(images=image, text=caption, padding="max_length", truncation=True, max_length=60, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in encoding.items()}

# ==========================================
# 4. 학습 루프
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 요청하신대로 CPU 기반 설정 (GPU 가능시 cuda로 변경 권장)
processor = BlipProcessor.from_pretrained(MODEL_LOAD_PATH)
model = BlipForConditionalGeneration.from_pretrained(MODEL_LOAD_PATH).to(device)

dataset = Epoch10Dataset(TRAIN_IMG_DIR, REPORT_PATH, processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

model.train()
print(f"🚀 학습 시작: Epoch 4 베이스 + GPT 9:1 몰입 전략 (총 샘플: {len(dataset)})")

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
            gc.collect()

        if (i+1) % SAVE_STEPS == 0:
            temp_path = CHECKPOINT_DIR / f"epoch10_step_{i+1}"
            model.save_pretrained(temp_path)
            print(f"\n💾 중간 저장: {temp_path}")

# 최종 저장
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
model.save_pretrained(MODEL_SAVE_PATH)
processor.save_pretrained(MODEL_SAVE_PATH)
print(f"✨ 모든 학습 완료! 모델 저장됨: {MODEL_SAVE_PATH}")