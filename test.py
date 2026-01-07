import torch
from PIL import Image
from pathlib import Path
import random
from transformers import BlipProcessor, BlipForConditionalGeneration

# ==========================================
# 1. 설정 및 모델 로드
# ==========================================
BASE_DIR = Path(r"D:/아카이브.ver2").resolve()
MODEL_PATH = BASE_DIR / "checkpoints/epoch7_final_checkpoints/final_step_35000"
TEST_IMG_DIR = BASE_DIR / "test" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
processor = BlipProcessor.from_pretrained(MODEL_PATH)
model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
model.eval()

# 차단할 단어 (환각 현상이 자주 발생하는 단어들 추가)
# 'street', 'plan', 'area', 'benoa' 등을 추가하여 엉뚱한 배경 설명을 막습니다.
bad_words = [
    "featuring", "scene", "features", "distinctive", "distinct", 
    "thebenoa", "benoa", "street", "plan", "area", "space", "sizes",
    "indoors", "outdoor", "indoor", "captured", "angle", "measurements" # 환각 단어 추가
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

# 테스트용 샘플 (20장)
samples = random.sample(all_test_images, min(20, len(all_test_images)))

print(f"\n🧪 [Temp 0.9 + 환각 억제] 테스트 시작")
print("-" * 70)

# ==========================================
# 2. 추론 실행
# ==========================================
for i, (img_path, true_label) in enumerate(samples):
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model.generate(
                **inputs, 
                max_length=55,
                min_length=20,           # 너무 길어서 헛소리하지 않게 약간 줄임
                repetition_penalty=2.5,  # 반복 억제 더 강화
                bad_words_ids=bad_words_ids,
                do_sample=True,
                top_k=50,
                top_p=0.7,
                temperature=0.9          # 0.9
            )
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            # --- 후처리 로직 ---
            caption = caption.replace(" - ", " ")
            words = caption.split()
            unique_words = []
            for w in words:
                # 단어 길이가 2자 이하이면서 기호가 섞인 이상한 단어(itsish 등) 필터링 시도
                if len(w) > 2 or w.lower() in ['a', 'an', 'is', 'in', 'on', 'at']:
                    if not unique_words or w.lower() != unique_words[-1].lower():
                        unique_words.append(w)
            
            # 문장 끝이 어색한 단어로 끝나면 과감히 삭제
            stop_words = ['in', 'the', 'at', 'with', 'and', 'of', 'showing', 'its', 'from', 'to', 'for', 'by']
            while unique_words and unique_words[-1].lower() in stop_words:
                unique_words.pop()
            
            final_caption = " ".join(unique_words)
            if not final_caption.endswith('.'):
                final_caption += "."
            # ------------------

        print(f"[{i+1}] 파일명: {img_path.name}")
        print(f"    ✅ 실제 정답: {true_label}")
        print(f"    🤖 모델 답변: {final_caption}")
        print("-" * 70)
        
    except Exception as e:
        print(f"Error: {e}")

print("\n✨ 0.9 버전 결과가 1.0보다 정확한가요? (특히 배경 설명 부분)")