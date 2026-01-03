import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 기존 파일들에서 클래스 및 함수 불러오기
from data_loader import Vocabulary, get_loader
from model import CNNtoRNN

# 1. 고정 경로 설정
INF_CONFIG = {
    "vocab_path": "vocab.pkl",
    "img_dir": "./data/images/test",   
    "json_dir": "./data/captions/test.json", 
    "model_path": "./checkpoints/best_efficientnet_b0_gru_AdamW.pth", # 학습된 모델 경로로 수정
    
    "batch_size": 1,
    "num_workers": 0,
    "image_size": (224, 224),
    "norm_mean": (0.485, 0.456, 0.406),
    "norm_std": (0.229, 0.224, 0.225)
}

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# 2. 성능 지표 계산 함수 (BLEU 1~4 & CIDEr 스타일 점수)
def calculate_metrics(reference_list, candidate_list):
    """
    reference_list: [[word1, word2, ...]] -> 정답 문장 리스트
    candidate_list: [word1, word2, ... ] -> 모델이 생성한 문장 리스트
    """
    chencherry = SmoothingFunction()
    
    # BLEU Scores
    b1 = sentence_bleu(reference_list, candidate_list, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1)
    b2 = sentence_bleu(reference_list, candidate_list, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)
    b3 = sentence_bleu(reference_list, candidate_list, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1)
    b4 = sentence_bleu(reference_list, candidate_list, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
    
    # CIDEr-style Score: n-gram 일치도의 가중 평균 (간이 구현)
    cider = (b1 * 0.1) + (b2 * 0.2) + (b3 * 0.3) + (b4 * 0.4)
    # 실제 CIDEr는 TF-IDF를 사용하지만, 단일 문장 비교에서는 고차 n-gram 일치도에 가중치를 둠

    return [b1, b2, b3, b4, cider]

# 3. 캡션 생성 함수
def get_caption(image_tensor, model, vocab, max_length=20):
    model.eval()
    result_caption = []
    
    with torch.no_grad():
        features = model.encoderCNN(image_tensor)
        inputs = features.unsqueeze(1)
        states = None
        
        for _ in range(max_length):
            # model.py에서 수정된 self.decoderRNN.rnn을 사용
            hiddens, states = model.decoderRNN.rnn(inputs, states)
            outputs = model.decoderRNN.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            predicted_id = predicted.item()
            
            word = vocab.itos[predicted_id]
            
            if word == "<END>":
                break
            if word not in ["<START>", "<PAD>", "<UNK>"]:
                result_caption.append(word)
            
            inputs = model.decoderRNN.embed(predicted).unsqueeze(1)
            
    return result_caption

# 4. 평가 실행 함수
def run_evaluation():
    # A. 체크포인트 로드
    print(f"[INFO] 모델 로드 중: {INF_CONFIG['model_path']}")
    try:
        checkpoint = torch.load(INF_CONFIG["model_path"], map_location=DEVICE)
    except Exception as e:
        print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {e}")
        return

    # B. 단어 사전 로드
    with open(INF_CONFIG["vocab_path"], "rb") as f:
        vocab = pickle.load(f)

    # C. 저장된 변수를 기반으로 모델 동적 생성 (자동 하이퍼파라미터 주입)
    model = CNNtoRNN(
        embed_size=checkpoint.get('embed_size', 256), 
        hidden_size=checkpoint.get('hidden_size', 512), 
        vocab_size=checkpoint.get('vocab_size', len(vocab)), 
        num_layers=checkpoint.get('num_layers', 1),
        encoder_type=checkpoint.get('encoder_type', 'resnet50'),
        decoder_type=checkpoint.get('decoder_type', 'lstm')
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[INFO] 모델 구조 복원 완료: {checkpoint.get('encoder_type')} + {checkpoint.get('decoder_type')}")

    # D. 데이터 로더 준비
    try:
        test_loader, _ = get_loader({**INF_CONFIG, "shuffle": False})
    except Exception as e:
        print(f"[ERROR] 데이터 로드 중 오류 발생: {e}")
        return

    # E. 샘플 테스트 및 지표 계산
    data_iter = iter(test_loader)
    images, captions = next(data_iter) 
    images = images.to(DEVICE)

    # 1) 추론 (Generated)
    gen_caption_list = get_caption(images, model, vocab)
    gen_caption_str = " ".join(gen_caption_list)

    # 2) 정답 변환 (Ground Truth)
    targets = captions[0].tolist()
    ground_truth_list = [vocab.itos[i] for i in targets if vocab.itos[i] not in ["<START>", "<END>", "<PAD>"]]
    ground_truth_str = " ".join(ground_truth_list)

    # 3) 지표 계산
    metrics = calculate_metrics([ground_truth_list], gen_caption_list)

    # 결과 출력
    print("\n" + "="*65)
    print(f"모델: {checkpoint.get('encoder_type').upper()} + {checkpoint.get('decoder_type').upper()}")
    print("-" * 65)
    print(f"정답 (GT): {ground_truth_str}")
    print(f"예측 (PR): {gen_caption_str}")
    print("-" * 65)
    print(f"BLEU-1: {metrics[0]:.4f} | BLEU-2: {metrics[1]:.4f}")
    print(f"BLEU-3: {metrics[2]:.4f} | BLEU-4: {metrics[3]:.4f}")
    print(f"CIDEr (가중합): {metrics[4]:.4f}")
    print("="*65)

    # 시각화
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(INF_CONFIG["norm_mean"], INF_CONFIG["norm_std"])],
        std=[1/s for s in INF_CONFIG["norm_std"]]
    )
    img_vis = inv_normalize(images[0]).cpu().permute(1, 2, 0).numpy().clip(0, 1)
    
    plt.imshow(img_vis)
    plt.title(f"PR: {gen_caption_str}", fontsize=10)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    run_evaluation()