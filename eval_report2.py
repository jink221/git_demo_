import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# ==========================================
# 1. 설정 (경로는 질문자님이 다시 맞추시면 됩니다)
# ==========================================
BASE_DIR = Path(r"D:/아카이브.ver2").resolve()
MODEL_PATH = BASE_DIR / "checkpoints/blip_finetuned_epoch11" # 경로 확인!
TEST_IMG_DIR = BASE_DIR / "test"
REPORT_SAVE_PATH = BASE_DIR / "evaluation_report_epoch11.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained(MODEL_PATH)
model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
model.eval()

# 가비지 절단을 위한 패턴
CUT_PATTERNS = ['a photo', 'an image', 'the photo', 'a high', 'this is', 'with a a', 'of a a']

def run_evaluation():
    results = []
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    all_images = [f for f in TEST_IMG_DIR.rglob("*") if f.suffix in extensions]
    
    print(f"🚀 [INFO] 총 {len(all_images)}장 분석 시작...")

    for img_path in tqdm(all_images):
        ground_truth = img_path.parent.name.lower().replace("_", " ")
        
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_length=65,
                    min_length=12,           # 너무 짧으면 묘사가 안 되므로 12~15 권장
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.4,  # 반복/나열 방지 강화
                    temperature=0.7,         # GPT 문장의 정확도를 위해 0.6으로 하향
                    no_repeat_ngram_size=3
                )
                caption = processor.decode(outputs[0], skip_special_tokens=True).lower()

            # --- [후처리: 가비지 절단] ---
            words = caption.split()
            full_txt = " ".join(words)
            cutoff_idx = len(words)
            for pattern in CUT_PATTERNS:
                if pattern in full_txt:
                    pattern_start_idx = full_txt.find(pattern)
                    curr_pos = 0
                    for idx, w in enumerate(words):
                        curr_pos = full_txt.find(w, curr_pos)
                        if curr_pos >= pattern_start_idx:
                            cutoff_idx = min(cutoff_idx, idx)
                            break
                        curr_pos += len(w)
            
            final_words = words[:cutoff_idx]
            # 불용어 제거 마무리
            stop_words = ['in', 'the', 'at', 'with', 'and', 'of', 'showing', 'its', 'from', 'to']
            while final_words and final_words[-1] in stop_words:
                final_words.pop()
            
            final_caption = " ".join(final_words).strip()
            # ---------------------------

            # 정확도 판정 (후처리된 문장에 정답이 있는지)
            is_correct = 1 if ground_truth in final_caption else 0
            
            results.append({
                "filename": img_path.name,
                "class": ground_truth,
                "generated_caption": final_caption + ".",
                "is_correct": is_correct
            })
            
        except Exception as e:
            print(f"\n⚠️ Error {img_path.name}: {e}")

    # 결과 저장
    df = pd.DataFrame(results)
    total_acc = df['is_correct'].mean() * 100
    class_report = df.groupby('class')['is_correct'].mean() * 100
    class_report = class_report.sort_values(ascending=False)

    print(f"\n🚩 [최종 결과] 종합 정확도: {total_acc:.2f}%")
    print("\n📊 성능 하위 10개 클래스 (주의 깊게 보실 부분):")
    print(class_report.tail(10))
    
    df.to_csv(REPORT_SAVE_PATH, index=False, encoding='utf-8-sig')
    print(f"\n✨ 리포트 저장 완료: {REPORT_SAVE_PATH}")

if __name__ == "__main__":
    run_evaluation()