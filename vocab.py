import json
import pickle
import re
import os
import nltk
from collections import Counter
from pathlib import Path


# 캡션 파일들이 저장된 폴더 경로
CAPTION_DIR = Path("./data/captions")
SAVE_PATH = "vocab.pkl"

# NLTK 데이터 경로 설정
NLTK_DATA_PATH = "./nltk_data"
if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)
nltk.data.path.insert(0, NLTK_DATA_PATH)

# 필요한 NLTK 데이터 다운로드 (최초 1회)
for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}', paths=[NLTK_DATA_PATH])
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DATA_PATH)


# 단어 사전 클래스
class Vocabulary:
    def __init__(self, freq_threshold=2):
        # 0: 패딩, 1: 문장시작, 2: 문장종료, 3: 모르는단어
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        # 소문자 변환 후 토큰화
        return nltk.word_tokenize(text.lower())

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4 # 특수 토큰 이후 4번 인덱스부터 시작

        print(f"[*] {len(sentence_list)}개의 문장으로부터 단어 수집 중...")
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1

                # 빈도수 문턱값을 넘는 순간 사전에 등록
                if frequencies[word] == self.freq_threshold:
                    if word not in self.stoi:
                        self.stoi[word] = idx
                        self.itos[idx] = word
                        idx += 1

    def numericalize(self, text):
        # 텍스트를 인덱스 번호 리스트로 변환
        tokenized_text = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]

def save_vocab(vocab, path):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"[SUCCESS] 단어 사전을 {path}에 저장했습니다.")

def load_vocab(path):
    # 나중에 모델 학습 시 사전을 불러올 때 사용
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    # 1. 모든 JSON 파일에서 캡션 수집 (train, val, test 통합)
    json_files = list(CAPTION_DIR.glob("*.json"))
    
    if not json_files:
        print(f"[ERROR] {CAPTION_DIR} 폴더에 JSON 파일이 없습니다.")
    else:
        all_captions = []
        print(f"[*] 대상 파일: {[f.name for f in json_files]}")

        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # JSON 내 'caption' 필드 데이터 수집
                all_captions.extend([item['caption'] for item in data])

        # 2. 사전 구축 (빈도수 2회 미만 단어는 제외하여 노이즈 제거)
        vocab = Vocabulary(freq_threshold=2)
        vocab.build_vocabulary(all_captions)

        print("-" * 40)
        print(f"✅ 구축 완료!")
        print(f"✅ 총 문장 수: {len(all_captions):,}")
        print(f"✅ 등록된 유니크 단어 수: {len(vocab):,}")
        print("-" * 40)

        # 3. 사전 저장
        save_vocab(vocab, SAVE_PATH)

        # 4. 검증 테스트
        if all_captions:
            sample_text = all_captions[0]
            print(f"\n[TEST] 원문: {sample_text}")
            print(f"[TEST] 토큰화 결과: {vocab.tokenizer(sample_text)}")
            print(f"[TEST] 인덱스 변환: {vocab.numericalize(sample_text)}")