import os
import json
import re
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration


CONFIG = {
    "BATCH_SIZE": 16,
    "NUM_WORKERS": 8,
    "NUM_BEAMS": 1,
    "MAX_NEW_TOKENS": 40,
    "MIN_NEW_TOKENS": 8,
    "REPETITION_PENALTY": 2.0
}

IMAGES_NAME = "test"
IMG_ROOT = Path(f"./data/images/{IMAGES_NAME}")
OUT_DIR = Path("./data/captions")
OUT_PATH = OUT_DIR / f"{IMAGES_NAME}.json"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 제거할 접두어 리스트 변수화
BAD_PREFIXES = [
    r"^a photo of\s+", 
    r"^an image of\s+", 
    r"^this image shows\s+", 
    r"^there is\s+", 
    r"^there are\s+", 
    r"^picture of\s+"
]

# 제거할 사진 출처 및 광고성 패턴 리스트
JUNK_PATTERNS = [
    r"photo by.*", 
    r"getty images.*", 
    r"stock photo.*", 
    r"copyright.*", 
    r"for wildlife.*", 
    r"society uk.*", 
    r"ltd\.", 
    r"available at.*",
    r"images for.*",
    r"alamy.*"  # 추가적인 스톡 이미지 사이트 대응
]


# 고속 처리를 위한 데이터셋 클래스
class BlipDataset(Dataset):
    def __init__(self, root_path, processor):
        self.img_files = sorted([p for p in root_path.rglob("*") if p.suffix.lower() in IMG_EXTS])
        self.processor = processor

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        class_name = img_path.parent.name
        try:
            image = Image.open(img_path).convert("RGB")
            # 텍스트 프롬프트 생성
            clean_label = class_name.replace('_', ' ')
            prompt_prefix = "an" if clean_label[0] in 'aeiou' else "a"
            prompt = f"{prompt_prefix} {clean_label},"
            
            # CPU에서 미리 전처리 (병렬 처리의 핵심)
            inputs = self.processor(images=image, text=prompt, return_tensors="pt", padding='max_length', max_length=32, truncation=True)
            return inputs.pixel_values[0], inputs.input_ids[0], str(img_path), class_name
        except Exception as e:
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    pixels, ids, paths, classes = zip(*batch)
    return torch.stack(pixels), torch.stack(ids), list(paths), list(classes)


# 후처리 함수
import re

# 제거할 접두어 리스트 변수화
BAD_PREFIXES = [
    r"^a photo of\s+", 
    r"^an image of\s+", 
    r"^this image shows\s+", 
    r"^there is\s+", 
    r"^there are\s+", 
    r"^picture of\s+"
]

# 제거할 사진 출처 및 광고성 패턴 리스트
JUNK_PATTERNS = [
    r"photo by.*", 
    r"getty images.*", 
    r"stock photo.*", 
    r"copyright.*", 
    r"for wildlife.*", 
    r"society uk.*", 
    r"ltd\.", 
    r"available at.*",
    r"images for.*",
    r"alamy.*"  # 추가적인 스톡 이미지 사이트 대응
]

def normalize_caption(raw: str, class_name: str) -> str:
    # 1. 기본 청소: 줄바꿈 제거, 마침표 기준 첫 문장만 추출, 소문자화
    s = raw.split("\n")[0].split(".")[0].strip().lower()
    
    # 2. 사진 출처 및 저작권 관련 정크 문구 제거 (JUNK_PATTERNS 활용)
    for pattern in JUNK_PATTERNS:
        s = re.sub(pattern, "", s).strip()

    # 3. 불필요한 접두어 반복 제거 (BAD_PREFIXES 활용)
    changed = True
    while changed:
        before = s
        for p in BAD_PREFIXES:
            s = re.sub(p, "", s).strip()
        if before == s: 
            changed = False

    # 4. 문장 끝에 남은 지저분한 기호나 접속사 정리 (쉼표, 슬래시 등)
    s = re.sub(r"[,/]\s*$", "", s).strip()

    # 5. 너무 짧거나 비어있으면 클래스명으로 강제 복구 (Fallback)
    clean_class = class_name.replace('_', ' ')
    if len(s.split()) < 2:
        s = clean_class

    # 6. 관사 및 복수형 처리 (이미 숫자나 관사로 시작하는지 체크)
    determiners = ("a ", "an ", "the ", "two ", "three ", "many ", "several ", "some ", "group ")
    if not s.startswith(determiners):
        vowels = ('a', 'e', 'i', 'o', 'u')
        article = "an" if s[0] in vowels else "a"
        s = f"{article} {s}"

    # 7. 관사 중복 최종 제거 (예: a the -> a)
    s = re.sub(r"\b(a|an|the)\s+(a|an|the)\b", r"\1", s)
    
    # 8. 최종 보정: 첫 글자 대문자 및 마침표 추가
    if not s:
        s = clean_class
    
    out = s[0].upper() + s[1:]
    if not out.endswith("."): 
        out += "."
        
    return out

def simplify_repeated_mentions(caption, class_name):
    clean_class = class_name.replace('_', ' ').lower()
    if caption.lower().count(clean_class) > 1:
        parts = re.split(r',| and ', caption.lower().replace('.', ''))
        adjectives = [p.strip().replace(clean_class, '').replace('an ', '').replace('a ', '').strip() for p in parts if p.strip()]
        adjectives = list(dict.fromkeys([a for a in adjectives if a]))
        if adjectives:
            combined = " and ".join(adjectives) if len(adjectives) == 2 else ", ".join(adjectives[:-1]) + f" and {adjectives[-1]}"
            new_cap = f"{combined} {clean_class}"
        else: new_cap = clean_class
        article = "An" if new_cap[0].lower() in 'aeiou' else "A"
        return f"{article} {new_cap}."
    return caption


# 메인 실행부
def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[*] 모델 로딩 중...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    processor.tokenizer.padding_side = "left"
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device).eval()

    dataset = BlipDataset(IMG_ROOT, processor)
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG["BATCH_SIZE"], 
        num_workers=CONFIG["NUM_WORKERS"], 
        collate_fn=collate_fn,
        shuffle=False
    )

    results = []
    
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=f"Capturing {IMAGES_NAME}"):
            if batch is None: continue
            pixel_values, input_ids, paths, classes = batch
            
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)

            out_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=CONFIG["MAX_NEW_TOKENS"],
                min_new_tokens=CONFIG["MIN_NEW_TOKENS"],
                num_beams=CONFIG["NUM_BEAMS"],
                no_repeat_ngram_size=3,
                repetition_penalty=CONFIG["REPETITION_PENALTY"]
            )

            raw_captions = processor.batch_decode(out_ids, skip_special_tokens=True)

            for path, raw_cap, c_name in zip(paths, raw_captions, classes):
                cap = normalize_caption(raw_cap, c_name)
                cap = simplify_repeated_mentions(cap, c_name)
                results.append({
                    "image": os.path.relpath(path, IMG_ROOT),
                    "caption": cap,
                    "class": c_name,
                })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Saved to {OUT_PATH} (Total: {len(results)} images)")

if __name__ == "__main__":
    main()