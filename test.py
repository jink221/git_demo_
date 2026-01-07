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

# 환각 억제 단어 리스트
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

# 테스트용 샘플 (20장)
samples = random.sample(all_test_images, min(20, len(all_test_images)))

print(f"\n🧪 [자연어 생성 모드] 테스트 시작 (Prompt-Free)")
print("-" * 75)

# ==========================================
# 2. 추론 실행
# ==========================================
for i, (img_path, true_label) in enumerate(samples):
    try:
        image = Image.open(img_path).convert("RGB")
        
        # [핵심 수정] text 인자를 제거하여 모델이 스스로 문장을 시작하게 함
        prompt="a"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model.generate(
                **inputs, 
                max_length=50,
                min_length=15,           # 문장이 너무 짧아지지 않게 유지
                repetition_penalty=2.0,  # 반복 억제
                bad_words_ids=bad_words_ids,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.8          # 팀원들의 피드백을 반영해 0.8로 안정화
            )
            caption = processor.decode(out[0], skip_special_tokens=True)
            
            # --- 후처리 로직 (중복 제거 및 문장 마감) ---
            words = caption.split()
            unique_words = []
            for w in words:
                if not unique_words or w.lower() != unique_words[-1].lower():
                    unique_words.append(w)
            
            stop_words = ['in', 'the', 'at', 'with', 'and', 'of', 'showing', 'its', 'from', 'to', 'for', 'by']
            while unique_words and unique_words[-1].lower() in stop_words:
                unique_words.pop()
            
            final_caption = " ".join(unique_words)
            if not final_caption.endswith('.'):
                final_caption += "."
            # ------------------------------------------

        print(f"[{i+1}] 파일명: {img_path.name}")
        print(f"    ✅ 실제 정답: {true_label}")
        print(f"    🤖 모델 답변: {final_caption}")
        print("-" * 75)
        
    except Exception as e:
        print(f"Error: {e}")

print("\n✨ 이제 'a photo of' 없이 모델이 직접 생성한 문장이 출력됩니다.")