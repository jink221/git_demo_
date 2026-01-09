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

# 환각 억제 단어
bad_words = [
    "thebenoa", "benoa", "street", "plan", "area", "space", "sizes",
    "indoors", "outdoor", "indoor", "captured", "angle", "measurements"
]
bad_words_ids = [processor.tokenizer.encode(w, add_special_tokens=False) for w in bad_words]
overused_words = ["natural", "environment", "distinctive", "appearance", "features", "scene", "view", "photo"]
overused_token_ids = [processor.tokenizer.encode(w, add_special_tokens=False) for w in overused_words]

# 2. 기존 환각 억제 단어들과 합치기
total_bad_words_ids = bad_words_ids + overused_token_ids
# 이미지 수집
all_test_images = []
for class_dir in TEST_IMG_DIR.iterdir():
    if class_dir.is_dir():
        cls_name = class_dir.name.replace("_", " ")
        imgs = list(class_dir.glob("*"))
        for img in imgs:
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                all_test_images.append((img, cls_name))

samples = random.sample(all_test_images, min(10, len(all_test_images)))

print(f"\n🔥 [어휘력 확장 + 정제 모드] 테스트 시작")
print("-" * 75)

# ==========================================
# 2. 추론 및 개선된 후처리 실행
# ==========================================
for i, (img_path, true_label) in enumerate(samples):
    try:
        image = Image.open(img_path).convert("RGB")
        
        # 시작 프롬프트 무작위화 (단어 다양성 확보의 핵심)
        prompts = [
            f"a centered shot of {true_label} with",
            f"the color of this {true_label} is",
            f"this {true_label} looks very",
            f"an image of {true_label} located"
        ]
        prompt = random.choice(prompts)
        
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model.generate(
                **inputs, 
                max_length=65,
                min_length=15,                      # 가비지 단어 나열 방지를 위해 적절히 제한
                repetition_penalty=1.5, 
                #bad_words_ids=total_bad_words_ids,
                do_sample=True,
                top_k=50, 
                top_p=0.95, 
                temperature=0.8, 
                no_repeat_ngram_size=3
            )
            # 마침표 제거하고 텍스트만 먼저 가져옴
            caption = processor.decode(out[0], skip_special_tokens=True).replace('.', '')
            
            # --- [개선된 후처리 순서] ---
            # 1. 단어 단위로 분리
            words = caption.split()
            
            # 2. 중복 단어 제거 (연속된 중복만 제거)
            unique_words = []
            for w in words:
                if not unique_words or w.lower() != unique_words[-1].lower():
                    unique_words.append(w)
            
            # 3. 문장 끝에 오면 어색한 불용어(조사/전치사) 제거
            stop_words = ['in', 'the', 'at', 'with', 'and', 'of', 'showing', 'its', 'from', 'to', 'for', 'by', 'a', 'an']
            while unique_words and unique_words[-1].lower() in stop_words:
                unique_words.pop()
            
            # 4. 최대 15단어 내외로 문장을 깔끔하게 제한 (LSTM 학습 효율화)
            # 만약 단어 수가 너무 많아져서 가비지가 섞이는 걸 방지하기 위함
            unique_words = unique_words[:18] 

            # 5. 최종 결합 후 마침표 추가
            final_caption = " ".join(unique_words).strip()
            if final_caption:
                final_caption += "."
            # ------------------------------------------

        print(f"[{i+1}] 파일명: {img_path.name}")
        print(f"    ✅ 실제 정답: {true_label}")
        print(f"    🤖 모델 답변: {final_caption}")
        print("-" * 75)
        
    except Exception as e:
        print(f"Error: {e}")

print("\n✨ 정제된 고품질 문장 생성이 완료되었습니다.")