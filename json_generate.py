import torch
import json
from PIL import Image
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm

# ==========================================
# 1. 설정 및 모델 로드
# ==========================================
BASE_DIR = Path(r"D:/아카이브.ver2").resolve()
# 가장 마지막에 성공한 모델 경로로 설정하세요
MODEL_PATH = BASE_DIR / "checkpoints/blip_finetuned_epoch11" 

DATA_PATHS = {
    "train": BASE_DIR / "train",
    "val": BASE_DIR / "val",
    "test": BASE_DIR / "test" # 테스트 폴더도 포함하여 전체 캡셔닝
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
processor = BlipProcessor.from_pretrained(MODEL_PATH)
model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
model.eval()

# 가비지 절단 패턴 (우리가 보강했던 리스트)
CUT_PATTERNS = [
    'a photo', 'an image', 'the photo', 'a high', 'this is', 
    'with a a', 'of a a', 'showing a a', 'in a a', 'and a brown', 'and the'
]

# ==========================================
# 2. 고품질 인퍼런스 함수
# ==========================================
def process_folder(target_dir):
    if not target_dir.exists():
        print(f"⚠️ {target_dir} 경로가 존재하지 않아 건너뜁니다.")
        return []

    results = []
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(list(target_dir.glob(f"**/{ext}")))
    
    print(f"\n📂 {target_dir.name} 폴더 처리 시작 (총 {len(image_paths)}개)...")

    # tqdm을 사용하여 실시간 진행 상황 표시
    for img_path in tqdm(image_paths, desc=f"Processing {target_dir.name}"):
        try:
            image = Image.open(img_path).convert("RGB")
            # 폴더명에서 언더바 제거하여 정답 라벨 추출
            true_label = img_path.parent.name.replace("_", " ") 
            
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                out = model.generate(
                    **inputs, 
                    max_length=65,           # 길이를 65로 소폭 상향하여 상세 묘사 확보
                    min_length=15, 
                    repetition_penalty=1.4,  # 너무 높으면(2.0) 문장이 어색해져서 1.4 권장
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.7,         # 우리가 검증한 최적 온도
                    no_repeat_ngram_size=3   # 반복 단어 억제 강화
                )
                raw_caption = processor.decode(out[0], skip_special_tokens=True).lower()
                
                # --- [강력한 후처리 로직] ---
                words = raw_caption.split()
                full_txt = " ".join(words)
                cutoff_idx = len(words)

                # 패턴 발견 시 뒷부분 삭제
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

                # 끝부분이 불용어로 끝나는 것 방지
                stop_words = ['in', 'the', 'at', 'with', 'and', 'of', 'showing', 'its', 'from', 'to', 'a', 'an']
                while final_words and final_words[-1] in stop_words:
                    final_words.pop()
                
                final_caption = " ".join(final_words).strip()
                if final_caption:
                    final_caption += "."
                # ------------------------------

                results.append({
                    "image_path": str(img_path.relative_to(BASE_DIR)),
                    "label": true_label,
                    "caption": final_caption
                })

        except Exception as e:
            print(f"\n❌ 에러 발생 ({img_path.name}): {e}")
            
    return results

# ==========================================
# 3. 실행 및 저장
# ==========================================
for folder_key, folder_path in DATA_PATHS.items():
    folder_results = process_folder(folder_path)
    if folder_results:
        save_name = f"{folder_key}_final_captions.json"
        with open(BASE_DIR / save_name, "w", encoding="utf-8") as f:
            json.dump(folder_results, f, ensure_ascii=False, indent=4)
        print(f"✨ {folder_key} 완료! 저장됨: {save_name} ({len(folder_results)}개)")

print("\n" + "="*50)
print("🏁 모든 데이터셋의 고품질 캡셔닝 작업이 종료되었습니다.")