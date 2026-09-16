---
title: "02 · Autograd — 기울기의 자동화"
layout: default
nav_order: 5
parent: "PyTorch 도서"
description: "02 · Autograd — 기울기의 자동화"
---

# 02 · Autograd — 기울기의 자동화

> 핵심 질문: PyTorch는 어떻게 `y = x² + 2x`의 미분값을 손으로 안 쓰고 얻는가?
> 선행: 01장. 소요: 25분.

## 2.1 `requires_grad`가 스위치다

01장의 tensor에 `requires_grad=True`를 켜면, 그 tensor에서 파생된 **모든 계산이 기록**됩니다.
기록된 계산은 뒤에서 "이 출력에 대해 내 입력의 기울기는 얼마?" 라고 되물을 수 있습니다.

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x        # y = x² + 2x
y.backward()              # y에 대한 모든 리프의 기울기 계산
print("dy/dx at x=3 (2x+2) =", x.grad.item())
```

```text
dy/dx at x=3 (2x+2) = 8.0
```

손으로 미분하면 `dy/dx = 2x + 2`, `x=3`에서 `8`. 정확히 일치합니다.
`backward()`가 한 일은 이 **chain rule을 자동 적용**한 것입니다.

## 2.2 계산 그래프와 `.grad`의 생애

- `x`처럼 사용자가 만든 tensor를 **leaf**라 하고, `.grad`가 채워집니다.
- `y`처럼 연산으로 파생된 tensor는 `.grad`가 `None`입니다(리프가 아니라서).

```python
w = torch.tensor([[2.0]], requires_grad=True)     # leaf (학습 대상)
x = torch.tensor([[1.0], [2.0], [3.0]])
y_true = torch.tensor([[5.0], [7.0], [9.0]])

pred = x @ w                    # 비리프: .grad 없음
loss = ((pred - y_true) ** 2).mean()
print("loss =", loss.item())

loss.backward()                 # loss → pred → w 로 기울기 역전파
print("dLoss/dW =", w.grad.item())
print("new leaf.grad before backward:",
      torch.tensor(1.0, requires_grad=True).grad)
```

```text
loss = 9.0
dLoss/dW = -12.0
new leaf.grad before backward: None
```

### 이 숫자들을 손으로 검산하기

`W=2`일 때:
- `pred = x @ W = [2,4,6]`
- `pred - y_true = [2-5, 4-7, 6-9] = [-3,-3,-3]`
- `loss = mean(9,9,9) = 9.0` ✓
- `dLoss/dW = mean(2·(pred−y_true)·x) = mean(2·(−3)·[1,2,3]) = mean(−6,−12,−18) = −12.0` ✓

`grad`가 음수라는 건 **W를 Increase 시켜야 loss가 줄어든다**는 뜻입니다.
03장의 경사 하강법은 바로 `W ← W − lr·grad` 로 이 정보를 씁니다.
(기울기가 음수 → 빼면 양수 방향 이동 → W 증가.)

## 2.3 함정 #1: `.grad`는 누적된다

`backward()`를 매 반복 호출하면 `.grad`가 **초기화되지 않고 계속 더해집니다.**
이것이 `simple_classifier.py`가 매 루프 끝에서 `W.grad.zero_()`를 명시적으로 호출하는 이유입니다.

```python
a = torch.tensor(2.0, requires_grad=True)
for _ in range(3):
    (a ** 2).backward()          # 안 지우고 계속 backward
print("누적된 grad (초기화 없음):", a.grad.item())   # 4+4+4 = 12
```

```text
누적된 grad (초기화 없음): 12.0
```

한 번만 계산했다면 `grad = 2a = 4`여야 하지만 3번 누적돼 12가 됐습니다.
**학습 루프에서 `zero_grad()`를 빠뜨리면 기울기가 부풀어 올라 발산합니다.** 03장에서 다시 등장.

## 2.4 함정 #2: `no_grad()` — 갱신 자체를 추적 금지하기

가중치를 직접 갱신할 때(`W -= lr*W.grad`), 그 갱신 연산조차 그래프에 기록되면 안 됩니다.
그래서 `with torch.no_grad():` 블록 안에서 갱신합니다.

```python
w = torch.tensor(5.0, requires_grad=True)
with torch.no_grad():
    w -= 1.0                     # 추적을 피해서 값만 변경
print("no_grad 안에서 갱신 후 w =", w.item())
```

```text
no_grad 안에서 갱신 후 w = 4.0
```

> `no_grad()` 없이 `w -= 1.0`을 하면 PyTorch가 "운영체제가 inplace 연산을 추적된 tensor에
> 하려 한다"며 에러를 냅니다. **추적 중인 tensor의 값을 바꾸는 모든 자리(학습 갱신, 추론)에
> `no_grad()`가 필요**하다는 습관을 붙이세요. 04장 `nn` API는 이걸 내부에서 알아서 해줍니다.

## 2.5 원본 예제와 연결

`simple_classifier.py`의 학습 루프를 autograd 관점으로 다시 읽으면:

```python
for _ in range(100):
    y_pred = x @ W + b          # forward: 이 결과가 목표와 얼마나 다른가
    loss = ((y_pred - y_true) ** 2).mean()   # 스칼라 손실
    loss.backward()             # ← dLoss/dW, dLoss/db 계산 (이 장)
    with torch.no_grad():       # ← 갱신은 추적 외 (2.4)
        W -= 0.01 * W.grad      # ← 경사 하강법 (03장)
        b -= 0.01 * b.grad
        W.grad.zero_()          # ← 누적 방지 (2.3)
        b.grad.zero_()
```

이 루프가 **딥러닝 학습의 전부**입니다. 규모가 커져도(05, 06장) 이 다섯 단계는 그대로입니다.
03장은 가운데 두 단계(손실 정의와 갱신 규칙)를, 04장은 이 전체를 `nn`/`optim`이 대신하게 하는 법을 다룹니다.

## 정리

| 개념 | 포인트 |
|------|--------|
| `requires_grad=True` | 이 tensor부터 계산을 기록하라 |
| `loss.backward()` | 리프 `.grad`에 기울기 채움 (chain rule 자동) |
| leaf vs 비리프 | `.grad`는 leaf에만 채워짐 |
| 기울기 부호 | `grad` 음수 → 값을 키우면 loss 감소 |
| `.grad` 누적 | 매 반복 `zero_grad()` 필수 |
| `no_grad()` | 갱신·추론 연산은 그래프에 남기지 말 것 |

## 직접 해보기

1. `y = x**3` 에 대해 `x=2` 일 때 `.grad`는? 손으로 계산한 `3x²`과 맞는지 확인.
2. 2.3 코드의 루프 안에서 `a.grad.zero_()`를 추가하면 최종 `grad`가 어떻게 변하는지 실험.
3. `z = x*y + y**2` (x, y 모두 `requires_grad=True`)에서 `z.backward()` 후 `x.grad`, `y.grad`를
   손으로 구한 편미분(`∂z/∂x = y`, `∂z/∂y = x + 2y`)과 비교.

다음: 이렇게 얻은 기울기를 **어떻게 값 갱신에 쓰는가** — [03. Loss와 Optimizer](03_loss_optimizer.md).
