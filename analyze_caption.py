import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# 분석할 파일 경로
JSON_PATH = Path("./data/captions/train.json")

def analyze_caption_quality(json_path):
    if not json_path.exists():
        print(f"[ERROR] 파일을 찾을 수 없습니다: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    if total == 0:
        print("데이터가 비어 있습니다.")
        return

    word_counts = []
    class_stats = defaultdict(lambda: {"total": 0, "issues": 0})
    
    # 결함 유형 확장
    defects = {
        "repeated_words": 0,    # 동일 단어 연속 반복
        "logical_errors": 0,    # and or, 이중 관사 등
        "no_verb_structure": 0, # 동사/전치사 부재
        "too_short": 0,         # 단어 수 부족
        "class_name_missing": 0, # [추가] 캡션에 클래스명이 없음
        "junk_contained": 0     # [추가] Photo by, Getty 등 포함
    }

    verb_keywords = ["is", "are", "sitting", "standing", "flying", "on", "in", "with", "eating", "perched"]
    junk_keywords = ["photo by", "getty", "images", "stock", "copyright", "ltd", "available"]

    for item in data:
        cap = item.get('caption', '').strip()
        c_name = item.get('class', 'unknown').replace('_', ' ').lower()
        low_cap = cap.lower().replace('.', '')
        words = low_cap.split()
        
        class_stats[c_name]["total"] += 1
        has_issue = False
        current_issues = []

        # 1. 길이 체크
        word_counts.append(len(words))
        if len(words) < 5:
            defects["too_short"] += 1
            has_issue = True

        # 2. 정크 문구 체크 (BLIP 특화 결함)
        if any(j in low_cap for j in junk_keywords):
            defects["junk_contained"] += 1
            has_issue = True

        # 3. 클래스명 누락 체크
        if c_name not in low_cap:
            defects["class_name_missing"] += 1
            has_issue = True

        # 4. 논리/문법 오류 (이중 관사 추가)
        if re.search(r"\b(and|or)\s+(and|or)\b", low_cap) or ", ," in low_cap or re.search(r"\b(a|an|the)\s+(a|an|the)\b", low_cap):
            defects["logical_errors"] += 1
            has_issue = True

        # 5. 연속 단어 반복
        if any(words[i] == words[i+1] for i in range(len(words)-1)):
            defects["repeated_words"] += 1
            has_issue = True

        # 6. 동사 구조 체크
        if not any(v in low_cap for v in verb_keywords):
            defects["no_verb_structure"] += 1
            has_issue = True
            
        if has_issue:
            class_stats[c_name]["issues"] += 1

    # 결과 출력
    print(f"\n" + "="*60)
    print(f"📊 캡션 품질 상세 분석 리포트: {json_path.name}")
    print("="*60)
    print(f"✅ 전체 이미지 수: {total:10}")
    print(f"📝 평균 단어 길이: {sum(word_counts)/total:10.2f} 단어")
    print("-" * 60)
    
    print(f"⚠️ [유형별 결함 문장 비율]")
    for k, v in defects.items():
        percentage = (v / total) * 100
        print(f"   - {k:20}: {v:5} 건 ({percentage:5.1f}%)")
        
    print("-" * 60)
    print(f"🚨 [결함률이 높은 TOP 5 클래스]")
    worst_classes = sorted(
        class_stats.items(), 
        key=lambda x: x[1]["issues"] / x[1]["total"] if x[1]["total"] > 0 else 0, 
        reverse=True
    )[:5]
    
    for name, stat in worst_classes:
        fail_rate = (stat["issues"] / stat["total"]) * 100
        print(f"   - {name:22}: {fail_rate:5.1f}% ({stat['issues']}/{stat['total']})")
    
    print("=" * 60)
    print("💡 junk_contained가 높으면: normalize_caption의 JUNK_PATTERNS를 보강하세요.")
    print("💡 class_name_missing이 높으면: 프롬프트에 클래스명을 더 명확히 주입하세요.\n")

if __name__ == "__main__":
    analyze_caption_quality(JSON_PATH)