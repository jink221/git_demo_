import torch
from PIL import Image
from pathlib import Path
import random
from transformers import BlipProcessor, BlipForConditionalGeneration

# ==========================================
# 1. 설정 및 모델 로드
# ==========================================
BASE_DIR = Path(r"D:/아카이브.ver2").resolve()
MODEL_PATH = BASE_DIR / "checkpoints/blip_finetuned_epoch10"
TEST_IMG_DIR = BASE_DIR / "test" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
processor = BlipProcessor.from_pretrained(MODEL_PATH)
model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
model.eval()

# 환각 억제 단어 설정 (생성 시 적용)
bad_words = ["thebenoa", "benoa", "street", "plan", "area", "space", "sizes", "indoors", "outdoor", "indoor"]
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

samples = random.sample(all_test_images, min(10, len(all_test_images)))

print(f"\n🔥 [가비지 절단 + 고품질 모드] 테스트 시작")
print("-" * 75)

# ==========================================
# 2. 추론 및 개선된 후처리 실행
# ==========================================
for i, (img_path, true_label) in enumerate(samples):
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model.generate(
                **inputs, 
                max_length=65,           # 너무 길면 가비지가 생기므로 65로 제한
                min_length=12, 
                repetition_penalty=1.4,  # 반복 억제 강화
                do_sample=True,
                top_k=40, 
                top_p=0.9, 
                temperature=0.6,         # 창의성보다 정확도 위주
                no_repeat_ngram_size=3
            )
            
            # 마침표 제거하고 텍스트 추출
            raw_caption = processor.decode(out[0], skip_special_tokens=True).replace('.', '')
            words = raw_caption.split()

            # --- [개선된 후처리: 가비지 절단 로직] ---
            
            # 1. 문장이 다시 시작되는 패턴(절단 지점) 정의
            # 모델이 헛소리를 시작할 때 주로 쓰는 패턴들입니다.
            cut_patterns = [
                'a photo', 'an image', 'the photo', 'a high', 'this is', 
                'with a a', 'of a a', 'showing a a', 'in a a'
            ]
            
            # 전체 문장을 소문자로 변환하여 패턴 검색
            full_sentence_lower = " ".join(words).lower()
            cutoff_idx = len(words)

            for pattern in cut_patterns:
                if pattern in full_sentence_lower:
                    # 패턴이 시작되는 위치(인덱스) 찾기
                    pattern_start_idx = full_sentence_lower.find(pattern)
                    # 해당 위치가 몇 번째 단어인지 계산
                    current_len = 0
                    for idx, word in enumerate(words):
                        current_len = full_sentence_lower.find(word.lower(), current_len)
                        if current_len >= pattern_start_idx:
                            # 패턴 시작 전까지만 남김
                            cutoff_idx = min(cutoff_idx, idx)
                            break
                        current_len += len(word)

            # 절단 적용
            meaningful_words = words[:cutoff_idx]

            # 2. 중복 단어 제거 (연속된 중복)
            unique_words = []
            for w in meaningful_words:
                if not unique_words or w.lower() != unique_words[-1].lower():
                    unique_words.append(w)
            
            # 3. 문장 끝 어색한 불용어 제거
            stop_words = ['in', 'the', 'at', 'with', 'and', 'of', 'showing', 'its', 'from', 'to', 'for', 'by', 'a', 'an']
            while unique_words and unique_words[-1].lower() in stop_words:
                unique_words.pop()
            
            # 4. 최종 길이 제한 (가독성)
            unique_words = unique_words[:25] 

            # 5. 마침표 추가 및 최종 결합
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

print("\n✨ 가비지가 제거된 정제 문장 테스트가 완료되었습니다.")