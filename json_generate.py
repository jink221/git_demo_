import torch
import json
from PIL import Image
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration

# ==========================================
# 1. 설정 및 모델 로드
# ==========================================
BASE_DIR = Path(r"D:/아카이브.ver2").resolve()
MODEL_PATH = BASE_DIR / "checkpoints/epoch7_final_checkpoints/final_step_35000"

# 이미 나누어져 있는 폴더 경로 설정
DATA_PATHS = {
    "train": BASE_DIR / "train",
    "val": BASE_DIR / "val"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
processor = BlipProcessor.from_pretrained(MODEL_PATH)
model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
model.eval()

# 환각 억제 단어 리스트
bad_words = ["thebenoa", "benoa", "street", "plan", "area", "indoors", "outdoor", "captured", "angle"]
bad_words_ids = [processor.tokenizer.encode(w, add_special_tokens=False) for w in bad_words]

# ==========================================
# 2. 캡셔닝 및 데이터 구조화 함수
# ==========================================
def process_folder(target_dir):
    results = []
    # 하위 폴더(클래스 폴더) 순회
    image_paths = list(target_dir.glob("**/*.*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    print(f"📂 {target_dir.name} 폴더 처리 시작 (총 {len(image_paths)}개)...")

    for i, img_path in enumerate(image_paths):
        try:
            image = Image.open(img_path).convert("RGB")
            true_label = img_path.parent.name.replace("_", " ") # 폴더명을 라벨로 사용
            
            # 검증된 'a' 프롬프트 방식
            prompt = "a"
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                out = model.generate(
                    **inputs, 
                    max_length=50,
                    min_length=15, 
                    repetition_penalty=2.0,
                    bad_words_ids=bad_words_ids,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.8
                )
                caption = processor.decode(out[0], skip_special_tokens=True)
                
                # --- 후처리 (중복 제거 및 마감) ---
                words = caption.split()
                unique_words = []
                for w in words:
                    if not unique_words or w.lower() != unique_words[-1].lower():
                        unique_words.append(w)
                
                stop_words = ['in', 'the', 'at', 'with', 'and', 'of', 'showing', 'its', 'from', 'to']
                while unique_words and unique_words[-1].lower() in stop_words:
                    unique_words.pop()
                
                final_caption = " ".join(unique_words)
                if not final_caption.endswith('.'): final_caption += "."
                # ------------------------------

                results.append({
                    "image_path": str(img_path.relative_to(BASE_DIR)),
                    "label": true_label,
                    "caption": final_caption
                })

            if (i + 1) % 100 == 0:
                print(f"📊 {target_dir.name} 진행률: {i + 1}/{len(image_paths)}")

        except Exception as e:
            print(f"❌ 에러 발생 ({img_path.name}): {e}")
            
    return results

# ==========================================
# 3. 메인 실행 및 파일 저장
# ==========================================
# Train 폴더 처리
train_results = process_folder(DATA_PATHS["train"])
with open(BASE_DIR / "train.json", "w", encoding="utf-8") as f:
    json.dump(train_results, f, ensure_ascii=False, indent=4)

# Val 폴더 처리
val_results = process_folder(DATA_PATHS["val"])
with open(BASE_DIR / "val.json", "w", encoding="utf-8") as f:
    json.dump(val_results, f, ensure_ascii=False, indent=4)

print("-" * 75)
print(f"✨ 최종 작업 완료!")
print(f"📝 생성된 파일: {BASE_DIR}/train.json ({len(train_results)}개)")
print(f"📝 생성된 파일: {BASE_DIR}/val.json ({len(val_results)}개)")