import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# ==========================================
# 1. 경로 설정
# ==========================================
BASE_DIR = Path(r"D:/아카이브.ver2").resolve()
MODEL_PATH = BASE_DIR / "checkpoints/blip_finetuned_epoch9_final"
TEST_IMG_DIR = BASE_DIR / "test"
REPORT_SAVE_PATH = BASE_DIR / "evaluation_report_epoch9.csv"

# ==========================================
# 2. 장치 및 모델 로드
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📦 [INFO] {device} 장치로 로딩 중: {MODEL_PATH.name}")

processor = BlipProcessor.from_pretrained(MODEL_PATH)
model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
model.eval()

def run_evaluation():
    results = []
    # 이미지 파일 수집
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    all_images = [f for f in TEST_IMG_DIR.rglob("*") if f.suffix in extensions]
    
    print(f"🚀 [INFO] 총 {len(all_images)}장의 이미지를 분석하고 상세 묘사를 생성합니다...")

    for img_path in tqdm(all_images):
        # 실제 정답 (폴더명 추출)
        ground_truth = img_path.parent.name.lower().replace("_", " ")
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # [수정포인트] 모델이 입을 열게 만드는 프롬프트 유도
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_length=65,           # 캡션 최대 길이
                    min_length=20,           # [중요] 상세 묘사를 위해 최소 길이 강제
                    do_sample=True,          # 창의적인 묘사를 위해 샘플링 허용
                    top_p=0.9,              # 상위 확률 단어 위주 선택
                    repetition_penalty=1.3,  # "a photo of a photo of..." 반복 방지
                    temperature=0.8,         # 적절한 다양성 부여
                    num_beams=1              # do_sample=True일 때는 1이 속도가 빠릅니다
                )
                caption = processor.decode(outputs[0], skip_special_tokens=True).lower()

            # 정확도 판정: 생성된 문장에 정답 클래스명이 포함되어 있는지 확인
            is_correct = 1 if ground_truth in caption else 0
            
            results.append({
                "filename": img_path.name,
                "class": ground_truth,
                "generated_caption": caption,
                "is_correct": is_correct
            })
            
        except Exception as e:
            print(f"\n⚠️ Error processing {img_path.name}: {e}")

    # ==========================================
    # 3. 결과 요약 및 저장
    # ==========================================
    df = pd.DataFrame(results)
    
    # 전체 정확도 계산
    total_acc = df['is_correct'].mean() * 100
    
    # 클래스별 성적표 생성
    class_report = df.groupby('class')['is_correct'].mean() * 100
    class_report = class_report.sort_values(ascending=False)

    print("\n" + "="*60)
    print(f"🚩 [최종 결과] 모델 종합 정확도: {total_acc:.2f}%")
    print("="*60)
    print("\n📊 상위 10개 클래스 성능:")
    print(class_report.head(10))
    print("\n🔻 하위 10개 클래스 성능:")
    print(class_report.tail(10))
    
    # CSV 저장 (UTF-8-SIG로 저장해야 한글 깨짐 방지 및 엑셀 호환 가능)
    df.to_csv(REPORT_SAVE_PATH, index=False, encoding='utf-8-sig')
    print(f"\n✨ [성공] 상세 리포트 및 캡션 결과가 저장되었습니다:")
    print(f"   📂 {REPORT_SAVE_PATH}")

if __name__ == "__main__":
    run_evaluation()