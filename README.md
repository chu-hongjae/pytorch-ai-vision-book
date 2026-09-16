---
title: "PyTorch"
layout: default
nav_exclude: true
---
# PyTorch · Tensor에서 이미지 분류까지

> 📱 **폰으로 읽으려면 여기서 여세요 → [GitHub Pages 문서 사이트](https://chu-hongjae.github.io/pytorch-ai-vision-book/)**
>
> 이 화면은 저장소 안내문이에요. 본문은 위 링크에 정렬·검색·모바일 레이아웃까지 갖춰서 있습니다.

이 저장소는 `03_ai_vision`(로봇 비전 학습 단계)의 PyTorch 파트에서 출발했습니다.
학습 메모 한 장과 `simple_classifier.py` 한 편이었는데, 그것을 도서 2권으로 다시 썼어요.

## 두 권의 책

| | 책 | 누구에게 |
|---|---|---|
| 📕 | [PyTorch — Tensor에서 이미지 분류까지](book/README.md) | 딥러닝이 처음인 사람. tensor → autograd → loss/optimizer → `nn.Module` → 미니배치 → CNN |
| 📗 | [FE/BE 개발자를 위한 AI 전향 가이드](switch/README.md) | 코드는 이미 잘 쓰는데 AI 방향은 처음인 사람. 패러다임 전환, 지표의 함정, 추론 API 배포 |

개념이 처음이면 📕를, **이미 개발 경력이 있고 방향만 돌리는 중**이면 📗를 먼저 읽으세요.
📗는 📕의 문법을 전제하지 않습니다. 같은 실험을 관점만 바꿔 봅니다.

## 이 문서의 신뢰 규칙

- 책에 나오는 **모든 코드 블록과 수치는 이 머신에서 실제로 실행한 stdout**입니다. 만들어낸 숫자가 하나도 없어요.
- 검증 환경은 PyTorch **2.8.0+cpu** — GPU 없이 전부 돌아갑니다. `torchvision`·`matplotlib`·`pandas`가 없는 조건이라, 이미지 예제는 합성 텐서로 만들었습니다.
- 난수를 쓰는 예제는 전부 `manual_seed`로 고정했습니다. 같은 버전이면 같은 값이 나옵니다.
- 각 장의 코드는 `book/code/`, `switch/code/` 에 독립 실행 파일로 있습니다. 책을 읽다가 바로 돌려볼 수 있어요.

```bash
# 책과 똑같이 돌려보기
cd book/code   && python ch03_learning_rate_lab.py
cd switch/code && python ch05_traps.py
```

## 저장소 구조

```text
.
├── index.md                  # Pages 사이트 진입점
├── _config.yml               # just-the-docs (remote_theme)
├── book/                     # 📕 PyTorch 개념서 (10챕터 + code/ 4종)
├── switch/                   # 📗 FE/BE → AI 전향 가이드 (10챕터 + code/ 4종)
├── simple_classifier.py      # 모든 것의 출발점인 원본 예제 (4줄짜리 학습 루프)
└── README.md                 # 이 파일
```

`simple_classifier.py`는 일부러 손대지 않았어요. 📕책의 03장이 이 예제의 `b = 1.6827`이
왜 목표값 3에 못 미치는지(에포크·학습률 부족)로 출발하거든요. 원본이 남아 있어야 그 이야기가 성립합니다.

---

## 원본 학습 개요 (그대로 보관)

아래는 이 폴더에 원래 적혀 있던 노트입니다. 책의 근거 자료여서 지우지 않았습니다.

### 학습 목표
- 딥러닝 기본 개념 이해
- PyTorch tensor와 모델 구현 학습
- 이미지 분류 모델의 기본 구조 이해

### 추천 GitHub 자료
- PyTorch 공식 저장소: https://github.com/pytorch/pytorch
- PyTorch Examples: https://github.com/pytorch/examples
- Hugging Face Transformers: https://github.com/huggingface/transformers

### 예제 코드
- `simple_classifier.py`

### 실습 과제
1. tensor 기본 연산 실습
2. 간단한 선형 분류 모델 구현
3. MNIST 또는 간단한 이미지 데이터 학습

### 핵심 포인트
- tensor operations
- loss function
- optimizer
- model training

---

*상위 단계: [03. AI Vision](../README.md) · Physical AI 전체 로드맵: [physical-ai-study-summary.md](../../physical-ai-study-summary.md)*
