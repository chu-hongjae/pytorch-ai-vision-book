---
title: "05 · 실제 학습 루프 — 미니배치"
layout: default
nav_order: 8
parent: "PyTorch 도서"
description: "05 · 실제 학습 루프 — 미니배치·DataLoader·과적합"
---

# 05 · 실제 학습 루프 — 미니배치·DataLoader·과적합

> 핵심 질문: 학습 데이터를 한 번에 다 쓰지 않고 잘게 쪼개면(미니배치) 무엇이 달라지는가?
> 선행: 04장. 소요: 30분.

## 5.1 왜 batch를 쪼개는가

04장은 3개 샘플을 통째로 썼습니다. 실제 데이터는 수백만 개라 메모리에 한 번에 못 올리고,
매번 전부를 쓰는 것보다 **작은 무작위 조각의 평균 기울기**를 쓰는 게 빠르고 때로 더 좋습니다.

| 방식 | 1스텝마다 쓰는 데이터 | 특징 |
|------|---------------------|------|
| Full-batch GD | 전부 | 정확하지만 느림·메모리 |
| Mini-batch SGD (표준) | `batch_size`개 | 균형 |
| SGD (stochastic) | 1개 | 빠르지만 요란 |

`x=[1,2,3]` 예제를 **100개 샘플의 노이즈 있는 회귀**로 확장해 직접 봅니다.

## 5.2 DataLoader — 데이터를 흘려보내는 파이프

PyTorch는 `TensorDataset`( tensor 묶음 ) + `DataLoader`( 배치로 잘라 순회 )를 제공합니다.

```python
import torch

torch.manual_seed(0)
N = 100
X = torch.rand(N, 1) * 10                    # x ∈ [0,10)
Y = 2 * X + 3 + torch.randn(N, 1) * 0.5      # y = 2x+3 + 노이즈

model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
ds = torch.utils.data.TensorDataset(X, Y)
dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)

for epoch in range(300):
    total = 0.0
    for xb, yb in dl:                        # ① batch 순회
        optimizer.zero_grad()                # ② (2.3 누격 방지)
        loss = torch.nn.functional.mse_loss(model(xb), yb)
        loss.backward()
        optimizer.step()                     # ③ 04장 5단계가 batch마다 반복
        total += loss.item() * xb.size(0)
    if epoch in (0, 99, 299):
        print(f"epoch {epoch:>3}  avg train loss {total / N:.4f}")

print("W =", round(model.weight.item(), 3), " b =", round(model.bias.item(), 3))
```

실측:

```text
epoch   0  avg train loss 128.5118
epoch  99  avg train loss 0.2376
epoch 299  avg train loss 0.2193
W = 2.028  b = 3.194
```

`W→2, b→3`에 근접. 노이즈 때문에 loss가 0은 아니지만(잔차가 분산 ≈0.25) **평균 기울기가
목표에 수렴**합니다. `shuffle=True` 덕분에 매 에포크 batch 구성이 바뀌어 local minimum에 안 갇힙니다.

- `for xb, yb in dl` 한 바퀴 = **1 에포크**.
- `loss = ...; total += loss.item()*len` 은 batch마다 크기가 달라 **개수 가중 평균**을 낸 것입니다.
- `torch.nn.functional.mse_loss`(함형 API)는 `MSELoss()` 객체와 같은 일. `nn.functional.*`은
  "층 없이 함수만" 쓰고 싶을 때 씁니다.

## 5.3 5단계가 batch마다 돌아간다

에포크 내부 구조를 펼치면:

```text
for epoch:                       # 전체 데이터를 몇 번 볼 것인지
    for xb, yb in dl:            #   mini-batch씩
        zero_grad → forward → loss → backward → step
```

즉 **에포크 = batch 루프의 횟수**, **스텝(step) = batch 하나 처리**.
데이터 100·batch 16 → 에포크당 약 7 스텝. 학습률을 논할 때 "스텝" 기준임을 혼동하지 마세요.

## 5.4 과적합 — train loss만 보면 빠지는 함정

합성 3클러스터 데이터로 MLP 분류기를 만들어보면, train loss는 곧 0이 됩니다.

```python
import torch

torch.manual_seed(0)
centers = torch.randn(3, 2) * 4
X, Y = [], []
for ci in range(3):
    X.append(centers[ci] + torch.randn(50, 2))
    Y.append(torch.full((50,), ci, dtype=torch.long))
X = torch.cat(X)                    # (150, 2)
Y = torch.cat(Y)                    # (150,)  int64 (클래스 레이블)

clf = torch.nn.Sequential(
    torch.nn.Linear(2, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3)
)
opt = torch.optim.Adam(clf.parameters(), lr=0.05)
lossf = torch.nn.CrossEntropyLoss()

for epoch in range(100):
    opt.zero_grad()
    logits = clf(X)                 # (150, 3) 원-hot 점수
    loss = lossf(logits, Y)         # Y는 int64 클래스 index 기대
    loss.backward()
    opt.step()
    if epoch in (0, 49, 99):
        acc = (logits.argmax(1) == Y).float().mean().item()
        print(f"epoch {epoch:>3}  loss {loss.item():.4f}  acc {acc:.2%}")
```

실측:

```text
epoch   0  loss 1.8487  acc 0.00%
epoch  49  loss 0.0000  acc 100.00%
epoch  99  loss 0.0000  acc 100.00%
```

**train 정확도 100%, loss 0** — 그런데 이걸 "좋다"고 할 수 있을까요?
3개의 잘 분리된 blob이라 모델이 **외울 수** 있습니다. 실제 배포 데이터(노이즈·미지의 위치)에서는
이렇게까지 안 맞습니다. **train만 보고 판단하면 과적합을 모릅니다.**

### CrossEntropy 약정 2가지 (오류의 90%가 이것)
1. **logit은 raw 점수** — `softmax`를 직접 치면 안 됩니다. `CrossEntropyLoss`가 내부에서
   `log_softmax + NLL` 을 수치 안정적으로 합쳐 처리합니다.
2. **레이블 Y는 int64 클래스 인덱스** (`[0,1,2,...]`) — one-hot을 기대하지 않습니다.
   float/one-hot을 주면 shape 에러. 04장 분류 헤드에서 `argmax`로 예측 클래스를 얻는 이유도 이것.

> 과적합을 다루려면 **train/test split**과 **검증 loss 모니터링**(train은 계속 줄지만 val이 오르기
> 시작하는 지점 = early stop)이 필요합니다. 이건 부록 A의 "확인해야 할 다음 개념"으로 넘깁니다.

## 5.5 모델 저장·복원 (state_dict) — 한 장의 실용

학습된 모델을 파일로 남기고 다시 씁니다. 06장·프로젝트에 필수.

```python
import torch
# clf는 위에서 학습된 상태라 가정
path = "model.pt"
torch.save(clf.state_dict(), path)          # 파라미터 사전만 저장

fresh = torch.nn.Sequential(
    torch.nn.Linear(2, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3)
)
fresh.load_state_dict(torch.load(path))     # 구조가 같아야 함
fresh.eval()                                # 추론 모드 (드롭아웃/BN 끔)
with torch.no_grad():                       # 2.4: 추론도 추적 불필요
    pred = fresh(X[:1]).argmax(1).item()
```

이 코드의 실측:

```text
loaded model predicts class for first blob point: 0 (true: 0 )
```

- `state_dict()`만 저장하는 이유: **가장 이식성 높고 안전**(pickle 전체 객체 저장은 금지 권장).
- `eval()` vs `train()`: 드롭아웃·BatchNorm이 학습/추론에서 다르게 동작. 추론 전 `eval()` 필수.
- 추론에는 `no_grad()` 또는 `torch.inference_mode()`: 기울기 불필요 → 메모리·속도 절약.

## 정리

1. 데이터를 잘게 **shuffle**해 batch마다 5단계 → 04장의 스케일아웃판.
2. **에포크 = 전체 1순회, 스텝 = batch 1처리.** lr·스텝 단위 혼동 금지.
3. train loss 0은 과적합일 수 있다. **검증 집합**이 없으면 일반화를 모른다.
4. 분류는 `CrossEntropyLoss`: raw logits + int64 index. `logits.argmax`로 예측.
5. `save(state_dict)` → `load_state_dict` + `eval()` + `no_grad()`가 실전 저장·추론 3종.

## 직접 해보기

`book/code/ch05_minibatch.py`:
1. `batch_size`를 4와 64로 바꿔보며 학습 곡수가 얼마나 매끄러워지는/거친지 관찰하세요.
2. 5.4의 학습 데이터에서 blob 간 거리를 줄여(중복되게) 바꾸면, test 없이도 loss가 안 떨어지는 걸 볼 수 있나요?
3. `shuffle=False`로 두고 학습이 어떻게 달라지는지 비교하세요(특정 순서에 치우침).

다음: 2D 점이 아니라 **픽셀 격자(이미지)** 를 인식하는 층 — [06장 CNN](06_cnn_vision.md).
