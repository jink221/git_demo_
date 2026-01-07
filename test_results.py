import torch
from PIL import Image
from pathlib import Path
import json
from transformers import BlipProcessor, BlipForConditionalGeneration

# ==========================================
# 1. 설정 및 모델 로드
# ==========================================
BASE_DIR = Path(r"D:/아카이브.ver2").resolve()
MODEL_PATH = BASE_DIR / "checkpoints/epoch7_final_checkpoints/final_step_35000"
TEST_IMG_DIR = BASE_DIR / "test" 
SAVE_JSON_PATH = BASE_DIR / "test_results.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
processor = BlipProcessor.from_pretrained(MODEL_PATH)
model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
model.eval()

# 환각 방지어 (테스트 시 효과가 좋았던 단어들)
bad_words = [
    "featuring", "scene", "features", "distinctive", "distinct", 
    "thebenoa", "benoa", "street", "plan", "area", "space", "sizes",
    "indoors", "outdoor", "indoor", "captured", "angle", "measurements"
]
bad_words_ids = [processor.tokenizer.encode(w, add_special_tokens=False) for w in bad_words]

# 이미지 수집
all_test_images = []
for class_dir in TEST_IMG_DIR.iterdir():
    if class_dir.is_dir():
        cls_name = class_dir.name.replace("_", " ")
        imgs = list(class_dir.glob("*"))
        for img in imgs:
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                all_test_images.append((img, cls_name))

# 결과를 담을 리스트
submission_data = []

print(f"✅ 총 {len(all_test_images)}개 이미지 처리 시작 (JSON 생성)")
print("-" * 70)

# ==========================================
# 2. 추론 및 데이터 구조화
# ==========================================
for i, (img_path, true_label) in enumerate(all_test_images):
    try:
        image = Image.open(img_path).convert("RGB")
        prompt = f"a photo of a {true_label} showing its" 
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model.generate(
                **inputs, 
                max_length=55,
                min_length=20,
                repetition_penalty=2.5,
                bad_words_ids=bad_words_ids,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.9
            )
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            # --- 후처리 로직 (테스트 검증 완료된 버전) ---
            caption = caption.replace(" - ", " ")
            words = caption.split()
            unique_words = []
            for w in words:
                if len(w) > 2 or w.lower() in ['a', 'an', 'is', 'in', 'on', 'at']:
                    if not unique_words or w.lower() != unique_words[-1].lower():
                        unique_words.append(w)
            
            stop_words = ['in', 'the', 'at', 'with', 'and', 'of', 'showing', 'its', 'from', 'to', 'for', 'by']
            while unique_words and unique_words[-1].lower() in stop_words:
                unique_words.pop()
            
            final_caption = " ".join(unique_words)
            if not final_caption.endswith('.'):
                final_caption += "."
            # ------------------------------------------

            # JSON 규격 생성
            submission_data.append({
                "file_name": img_path.name,
                "class": true_label,
                "model_answer": final_caption
            })
            
            if (i+1) % 100 == 0:
                print(f"📊 진행 상황: {i+1}/{len(all_test_images)} 완료")

    except Exception as e:
        print(f"❌ Error at {img_path.name}: {e}")

# ==========================================
# 3. JSON 파일 저장
# ==========================================
with open(SAVE_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(submission_data, f, ensure_ascii=False, indent=4)

print("-" * 70)
print(f"✨ 최종 JSON 생성 완료!")
print(f"📁 경로: {SAVE_JSON_PATH}")