import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import random
import gc
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration

# ==========================================
# 1. 설정 및 경로 (D드라이브 & 최신 성적표 반영)
# ==========================================
BASE_DIR = Path(r"D:/아카이브.ver2").resolve()
# [수정] 가장 최신 성적표인 10000 step 결과를 사용합니다.
REPORT_PATH = BASE_DIR / "evaluation_report_epoch7_10000.csv" 

# [수정] 온전하게 저장된 step 10000 체크포인트 로드
MODEL_LOAD_PATH = BASE_DIR / "checkpoints" / "epoch7_checkpoints" / "step_resume_10000"
# 결과 저장 경로
MODEL_SAVE_PATH = BASE_DIR / "checkpoints" / "blip_finetuned_epoch7_final"
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "epoch7_final_checkpoints"
TRAIN_IMG_DIR = BASE_DIR / "train"

LEARNING_RATE = 5e-8  # 아주 미세하게 조정하여 정확도를 깎지 않도록 함
BATCH_SIZE = 1 
EPOCHS = 1
SAVE_STEPS = 5000
TARGET_ACCURACY = 96.5 # 목표치를 조금 더 높게 설정

# ==========================================
# 2. 묘사 유도형 템플릿 (환각 방지용)
# ==========================================
def get_final_style_caption(cls_folder_name):
    clean_name = cls_folder_name.replace("_", " ")
    # 억지스러운 묘사 대신, 모델이 사진의 '특징'에 집중하게 만드는 문구들
    templates = [
        f"a photo of a {clean_name} showing its distinct features",
        f"a clear view of the {clean_name} in its natural environment",
        f"this image captures the details of a {clean_name}",
        f"a detailed shot of a {clean_name} from a specific angle",
        f"the {clean_name} is clearly visible in this scene"
    ]
    return random.choice(templates)

# ==========================================
# 3. 데이터셋 (최신 성적표 기반 집중 학습)
# ==========================================
class FinalRefineDataset(Dataset):
    def __init__(self, img_dir, report_path, processor):
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.samples = []

        # 최신 성적표 로드
        df = pd.read_csv(report_path)
        class_scores = df.groupby('class')['is_correct'].mean() * 100
        
        # 성적이 낮은 순으로 정렬하여 출력 (확인용)
        weak_classes = class_scores[class_scores < TARGET_ACCURACY].sort_values().index.tolist()
        print(f"\n🎯 최종 교정 타겟 클래스: {len(weak_classes)}개")
        print(f"📉 최저점 클래스 예시: {weak_classes[:5]}")

        for class_dir in tqdm(self.img_dir.iterdir(), desc="데이터 리스트 생성"):
            if class_dir.is_dir():
                cls_folder_name = class_dir.name
                cls_display_name = cls_folder_name.replace("_", " ")

                # [가중치 전략] 
                # 1. 90% 미만: 20배 폭격 (Bear 등)
                # 2. 90%~96%: 10배 집중 교정
                # 3. 나머지: 1배 (복습)
                score = class_scores.get(cls_display_name, 100)
                if score < 90.0:
                    repeat = 20
                elif score < TARGET_ACCURACY:
                    repeat = 10
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
        caption = get_final_style_caption(cls_folder_name)
        with Image.open(img_path) as img:
            image = img.convert("RGB")
            # 50% 확률로 텍스트 힌트 없이 이미지로만 학습 (시각 지능 강화)
            use_text = caption if random.random() > 0.5 else ""
            encoding = self.processor(images=image, text=use_text, padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in encoding.items()}

# ==========================================
# 4. 학습 실행
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained(MODEL_LOAD_PATH)
model = BlipForConditionalGeneration.from_pretrained(MODEL_LOAD_PATH).to(device)

dataset = FinalRefineDataset(TRAIN_IMG_DIR, REPORT_PATH, processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

scaler = torch.cuda.amp.GradScaler()
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

model.train()
print(f"\n🚀 step 10,000부터 마지막 교정 학습을 시작합니다.")

for epoch in range(EPOCHS):
    pbar = tqdm(dataloader, desc="Final Refining")
    for i, batch in enumerate(pbar):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            # 텍스트가 빈 경우(use_text="") labels를 어떻게 줄지 모델이 학습함
            loss = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids).loss
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if (i+1) % 100 == 0:
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'step': i+1})

        if (i+1) % SAVE_STEPS == 0:
            checkpoint_path = CHECKPOINT_DIR / f"final_step_{i+1}"
            model.save_pretrained(checkpoint_path)
            processor.save_pretrained(checkpoint_path)

        if i % 100 == 0: gc.collect()

MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
model.save_pretrained(MODEL_SAVE_PATH)
processor.save_pretrained(MODEL_SAVE_PATH)
print(f"\n✨ 모든 과정이 완료되었습니다! 수고하셨습니다.")