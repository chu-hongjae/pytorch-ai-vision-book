---
title: "부록 B · 용어 대조 · 생태계 · FAQ"
layout: default
nav_order: 29
parent: "FE/BE → AI 전향 가이드"
description: "개발자 관점 용어 사전, 도구 생태계 지도, 전향자 FAQ"
---
# 부록 B · 용어 대조표 · 생태계 · FAQ

## B.1 용어 대조 (개발자 → AI)

| AI 용어 | English | 한 줄 정의 | 개발자 관점 번역 |
|---------|---------|------------|------------------|
| 텐서 | tensor | 숫자 격자 + 연산/autograd | dtype·shape을 가진 다차원 배열 |
| 스칼라 손실 | loss | 틀린 정도의 단일 숫자 | 회귀 테스트의 diff 점수 |
| 기울기 | gradient | loss의 파라미터 미분 | 파라미터별 "여기 틀림"의 기여도 |
| 경사 하강 | gradient descent | `w -= lr·grad` 반복 | 원인 분해 후 상태 패치 |
| 학습률 | learning rate | 걸음 크기 (03📕에서 발산 실측) | 배치 크기 같은 운영 파라미터 |
| 옵티마이저 | optimizer | 갱신 규칙(SGD/Adam) | 마이그레이션 실행 전략 |
| 파라미터 | parameters | 학습으로 바뀌는 값 | 런타임 설정이 아니라 **데이터** |
| 하이퍼파라미터 | hyperparameters | 사람이 정하는 값 | `.env` / config |
| 층 | layer | 변환 한 칸 | 미들웨어·인터셉터 체인 |
| 순전파 | forward | 입력→출력 계산 | 함수 호출 |
| 역전파 | backward | 기여도 계산 | 스택을 타고 원인을 거꾸로 추적 |
| 에포크 | epoch | 전체 데이터 1순회 | 배치 잡 1사이클 |
| 배치 | batch | 한 스텝의 묶음 | DB 벌크 insert 단위 |
| 체크포인트 | checkpoint | 스냅샷 | DB 백업 |
| state_dict | — | 파라미터 이름→tensor 사전 | 직렬화된 상태 맵 |
| 사전학습 | pretrained | 남이 맞춘 파라미터 | 레거시 라이브러리/포크 |
| 파인튜닝 | fine-tuning | 사전학습을 내 데이터로 조금 더 학습 | 레거시 커스텀 유지보수 |
| 임베딩 | embedding | 의미를 벡터로 | 해프하지만 거리가 의미인 키 |
| 토큰 | token | 텍스트 최소 단위 | 파싱된 lexer 토큰과 같은 개념 |
| 컨텍스트 창 | context window | 한 번에 보는 입력 길이 | 요청 페이로드 크기 제한 |
| 추론 | inference | 가중치 갱신 없는 계산 | 읽기 전용 엔드포인트 |
| 일반화 | generalization | unseen에서도 맞음 | 테스트 커버리지 밖 동작 |
| 과적합 | overfitting | train만 맞음 | 통과하는 빈 껍데기 테스트 |
| 과소적합 | underfitting | 표현력이 모자람 | 용량 부족 |
| 편향 | bias | 형태 오차로 생긴 천장(01📦 A2) | 설계 오류(설정으론 못 고침) |
| 분산 | variance | 데이터 흔들림에 민감 | flaky 테스트와 유사 |
| 누수 | leakage | 정답 정보가 입력에 섞임 | 출력값을 입력에 넣은 테스트 |
| 드리프트 | drift | 배포 후 입력 분포 이동 | 스키마 변경 없는 데이터 부패 |
| 혼동 행렬 | confusion matrix | TC/FP/FN/TN | 에러 분류 트리에 대응 |
| 재현율 | recall | 진짜 중 잡은 비율 (05.1: 0.4333) | 놓친 사고 비율의 보수 |
| 정밀도 | precision | 잡은 중 맞은 비율 | 오경보 비용의 보수 |
| 검증 집합 | val set | 튜닝용 unseen | staging |
| 테스트 집합 | test set | 최종 1회 unseen | production |

> 📦 = 📕 PyTorch 개념도서를 가리킵니다. 같은 실험의 다른 렌즈이므로 서로 참조해서 보세요.

## B.2 생태계 지도 — 무엇을 언제 쓰나

**읽기 순서 그대로, 나중에 필요할 때 보면 됩니다. 지금 설치하지 마세요.**

| 계층 | 도구 | 이 책에서의 위치 |
|------|------|------------------|
| 연산 | PyTorch, JAX | 📕 전체 |
| 데이터 | NumPy, Pandas, Polars, Arrow | 03장(Pandas 이 머신엔 없음 → numpy/sklearn로 대체) |
| 전통 ML | scikit-learn | 05·03장 기준선·지표·스플릿 |
| 비전 데이터 | torchvision, Albumentations | 📕06장 범위 밖(미설치) |
| 실험 추적 | Weights & Biases, MLflow, TensorBoard | 5.4 체크리스트를 자동화하는 도구 |
| 서빙 | FastAPI, TorchServe, ONNX Runtime, Triton | 06장(stdlib로 원리 확인) |
| 양자화·가속 | torch.compile, ONNX, TensorRT, OpenVINO | 6.4 지연 예산 |
| 로봇 연결 | ROS2, OpenCV, Isaac Sim/Gazebo | 3막·Phase 5 |
| LLM/어시스턴트 | HF Transformers, llama.cpp, vLLM | 04 이후 |

### 이 표에서 가져갈 전략

**"2개 계층에서 깊고 1개 계층에서 넓게."** 전향자 90일에는 서빙(06)과 데이터(03)가 그 2개입니다.
모델 계층은 📕책 분량이면 충분하고, 그 above/below가 당신의 기존 전문성과 이어집니다.

## B.3 FAQ — 전향자가 실제로 묻는 것

**Q1. 수학을 다시 배워야 하나요?**
필요한 건 3개입니다: ① 미분 = "어디가 내리막인가" ② 벡터 내적 = "유사도" ③ 기댓값·분산 = "지표가 흔들림".
이 책과 📕02·03장이 저 3개를 코드로만 다룹니다. 먼저 책 읽고, 막히면 그때 개념어를 검색하세요.
**역순이 실패 경로입니다.**

**Q2. 논문 읽기는 언제부터?**
7.2의 2막을 3바퀴 돈 다음. 그 전에 읽으면 인용 구문을 외우게 됩니다.
읽는 법: 수식 → 구현 → 실험표 순이 아니라, **실험 설정 표**부터 읽어 "이 결과가 몇 번째 데이터에서 나왔나"를 먼저 확인.

**Q3. CUDA/그래픽카드 사야 하나요?**
이 책과 📕책은 2.8.0+cpu에서 전부 실행됩니다(0.1장에서 검증). GPU가 필요한 순간은 2막 후반~3막.
그때는 Colab/Kaggle 무료 크레딧으로 충분하고, 구매 판단은 **3막에서 자기 워크로드를 재본 다음**에 하세요.

**Q4. LLM 시대에 CNN·torch를 배울 가치가 있나요?**
Physical AI가 목적이라면 **예, 필수**입니다. 이유는 3개:
① 로봇·임베디드·실시간에서는 크기·지연·전력 때문에 소형 CNN/전통 ML이 현역입니다.
② LLM API만 쓰는 사람과 "모델을 직접 학습·평가·배포할 수 있는 사람"은 직무가 다릅니다.
③ LLM이 못 하는 것(시각 위치 추정, 센서 이상 탐지, 제어)이 이 저장소 로드맵의 본진입니다.

**Q5. 기존 개발 경력을 이력서에서 지워야 하나요?**
아닙니다. 00장 0.3표가 그 이유이고, 06장 실측(병목이 HTTP 클라이언트 사용법이었다)이 그 증거입니다.
**"AI를 배포까지 할 수 있는 개발자"** 는 "모델을 아는 사람"보다 시장이 큽니다. 포지셔닝을 그렇게 하세요.

**Q6. 코드를 베껴도 되나요?**
이 책 코드 전체가 복붙용입니다. 대신 **한 줄을 고쳐서 결과가 바뀌는 실험**을 1회 이상 하세요.
베끼기만 하면 04장의 `eval()` 누락 같은 사고를 디버깅할 수 없습니다.

**Q7. 언제 다른 걸로 갈아타야 하나요(JAX/TF 등)?**
지금으로서는 없습니다. 학습 개념(경사·손실·평가·서빙)은 1:1로 이전됩니다.
직장이 특정 스택을 요구하면 그때 옮기면 1주일면 됩니다.

## B.4 빠른 참조 — 자주 쓰는 스니펫

```python
import torch

torch.manual_seed(0)                              # 재현성: 모든 예제의 첫 줄

# 학습 루프 5단계
for epoch in range(N):
    for xb, yb in dl:
        opt.zero_grad(); loss = criterion(model(xb), yb)
        loss.backward(); opt.step()

# 추론 3종 세트 (04·06장)
model.eval()
with torch.no_grad():
    logits = model(x)
prob = torch.softmax(logits, dim=1)               # raw 점수 → 확률

# 저장 / 로드 (가중치만, 코드와 분리)
torch.save(model.state_dict(), "m.pt")
model.load_state_dict(torch.load("m.pt", weights_only=True))

# shape 디버깅 (가장 많이 씀)
print(x.shape, x.dtype, x.device)                 # 세 개를 함께 출력

# 지표 기본기 (05장)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
```

## B.5 이 책의 모든 실측 값 (한 표)

| 장 | 실험 | 값 |
|----|------|-----|
| 01 | 원형 경계, 손으로 쓴 축 임계값 | 0.8048 (= 항상-바깥 상수함수 0.80475) |
| 01 | 피처 `x²+y²` + 임계값 한 줄 | **1.0000** |
| 01 | 손으로 쓴 L1 다이아몬드 최선 | 0.9655 (형태 천장) |
| 01 | MLP 피팅 (train/test) | 0.9996 / **0.9975** |
| 03 | 타깃 누수 포함 test acc | **1.0000** (정상 0.8333, 항상-기준선 0.8022) |
| 04 | Dropout `train()` 2회 호출 | 출력 상이 → 비결정 |
| 04 | `eval()` 2회 호출 | 출력 동일 → 결정 |
| 05 | 불균형 model accuracy | 0.9633 |
| 05 | 항상-0 상수함수 accuracy | 0.9500 (**격차 1.33%p**) |
| 05 | 소수 클래스 재현율 | 0.4333 (17/30 놓침) |
| 05 | 과적합 (hidden=32) train/val MSE | 0.05282 / **0.57437** |
| 05 | 과대규제 (wd=0.02) train/val MSE | 0.10158 / 0.62204 (악화) |
| 06 | 순수 추론 | 0.022 ms/req |
| 06 | HTTP 왕복 (클라이언트 재사용) | 1.75 ms/req |
| 06 | HTTP 왕복 (매번 새 클라이언트) | **328.00 ms/req** |
| 06 | 서빙 모델 test acc | 0.9983 |

> 전 값: PyTorch 2.8.0+cpu / scikit-learn / numpy, 이 머신에서 실행한 stdout입니다.
> 📕책의 실측값은 [부록 A](../book/appendix_cheatsheet.md) 참조.
