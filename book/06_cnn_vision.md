---
title: "06 · 이미지 분류 — CNN"
layout: default
nav_order: 9
parent: "PyTorch 도서"
description: "06 · 이미지 분류 — CNN의 기본 구조"
---

# 06 · 이미지 분류 — CNN의 기본 구조

> 핵심 질문: 픽셀 격자를 입력받아 "이 이미지엔 무엇이 있는가"를 답하는 모델은 어떻게 생기나?
> 선행: 04, 05장. 소요: 35분. 로봇 비전(`03_ai_vision` 목표)으로의 교두보.

## 6.1 이미지는 `(채널, 세로, 가로)` tensor

01장 감각으로 돌아가, 이미지를 tensor로 보면 `(C, H, W)` shape의 숫자 격자입니다.
- 흑백 8×8 → `(1, 8, 8)`, RGB 64×64 → `(3, 64, 64)`.
- batch를 붙이면 `(N, C, H, W)`. PyTorch의 `Conv2d`는 정확히 **4D**를 기대합니다.

로봇 카메라 프레임 한 장도 `cv2.imread`가 `(H, W, 3)` (OpenCV는 BGR·채널 순서 마지막)으로 주고,
PyTorch에 넣으려면 `(N, 3, H, W)`로 **채널을 앞으로, batch 축 추가**를 해야 합니다.
`permute(0,3,1,2)` 같은 변환이 여기서 나옵니다(`opencv/image_basic.py`와 연결).

## 6.2 Fully-connected로는 왜 안 되는가

`Linear`로 28×28 이미지를 바로 받으려면 784개 픽셀을 납작하게 펴야 합니다. 문제는:
- 인접 픽셀의 **공간 관계(패턴·모서리·획)** 이 평면화되면 사라집니다.
- 픽셀 위치가 1칸만 어긋나도 입력 벡터가 완전히 달라져 **위치 민감**합니다.
- 파라미터가 폭증(28×28→100000급)합니다.

**CNN은 "작은 창(커널)을 훑으며 지역 패턴을 추출"** 함으로써 이걸 해결합니다.
합성 연산(01장)이 위치와 무관한 필터를 만드는 식입니다.

## 6.3 Conv2d — 픽셀 격자 위의 sliding window

```python
import torch

conv = torch.nn.Conv2d(
    in_channels=1, out_channels=4, kernel_size=3, padding=1
)
img = torch.randn(2, 1, 8, 8)          # (batch 2, C1, 8, 8)
feat = conv(img)                        # 4개 필터로 스캔
print("in :", tuple(img.shape))
print("out:", tuple(feat.shape))
print("kernel tensors:",
      sum(p.numel() for p in conv.parameters()))
```

```text
in : (2, 1, 8, 8)
out: (2, 4, 8, 8)
kernel tensors: 40
```

- `out_channels=4` → **4개의 서로 다른 3×3 필터**가 각 채널에 하나의 특성맵을 냅니다.
- `padding=1` 때문에 가로세로 8이 유지됐습니다. 커널 3×3에 한 칸 채움 = "주변 1픽셀을 봐도 크기 보존".
  `out_hw = (in_hw + 2·padding − kernel)/stride + 1 = (8+2−3)/1+1 = 8`.
- 파라미터 수: 필터가 4개, 각각 `1·3·3=9` 가중치 + bias 1 = `(9+1)·4 = 40`. **입력 해상도와 무관**
  한 게 핵심 — 커널이 위치마다 공유됩니다(weight sharing).

> `Conv2d`는 "채널 방향을 축으로 둔 지역 행렬곱"입니다. `Linear`가 위치를 무시하고 전부 연결하는
> 것과 달리, **가까운 픽셀만** 연결하고 그 연결을 전역에서 재사용합니다.

## 6.4 풀링 — 크기를 줄이며 강건하게

`MaxPool2d(2)`은 2×2 블록에서 **최댓값**만 남깁니다(해상도 절반).
- 위치가 좀 흔들려도 최대 반응은 유지 → **평행 이동 강건성**.
- 연산량·파라미터를 줄여 다음 층을 쉽게.

## 6.5 전체 분류기 — 합성 8×8 두 클래스로 검증

`03_ai_vision` 목표인 "이미지 분류 모델의 기본 구조"를 torchvision 없이 실현합니다.
**아래=밝음 vs 위=밝음** 을 위치로 구분하는 합성 이미지로 분류를 학습합니다.

```python
import torch

torch.manual_seed(0)

def make_toy(n_per=40):
    xs, ys = [], []
    for c in range(2):                       # 클래스 0/1
        base = torch.zeros(n_per, 1, 8, 8)
        if c == 0:
            base[:, :, 4:8, :] = 1.0          # 아래쪽이 밝음
        else:
            base[:, :, 0:4, :] = 1.0          # 위쪽이 밝음
        base += 0.1 * torch.randn_like(base)  # 노이즈
        xs.append(base); ys.append(torch.full((n_per,), c, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


X, Y = make_toy()
perm = torch.randperm(X.size(0))              # 샘플 순서 섞음 (RNG 상태에 영향)
X, Y = X[perm], Y[perm]
net = torch.nn.Sequential(
    torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),  # (N,1,8,8)->(N,4,8,8)
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),                             # ->(N,4,4,4)
    torch.nn.Flatten(),                                # ->(N,64)
    torch.nn.Linear(4 * 4 * 4, 2),                     # 분류 헤드 -> 2 classes
)
opt = torch.optim.Adam(net.parameters(), lr=0.01)
lossf = torch.nn.CrossEntropyLoss()

for epoch in range(30):
    opt.zero_grad()
    logits = net(X)
    loss = lossf(logits, Y)
    loss.backward()
    opt.step()
    if epoch in (0, 14, 29):
        acc = (logits.argmax(1) == Y).float().mean().item()
        print(f"epoch {epoch:>2}  loss {loss.item():.4f}  acc {acc:.1%}")
```

실측:

```text
epoch  0  loss 0.7321  acc 50.0%
epoch 14  loss 0.0069  acc 100.0%
epoch 29  loss 0.0002  acc 100.0%
```

`acc 50%`(랜스 추첨)에서 출발해 **100%** 로 수렴했습니다. 5단계 루프(04장)가 그대로고,
달라진 건 `model(X)`가 `Conv→ReLU→Pool→Flatten→Linear`가 됐다는 것뿐입니다.

## 6.6 CNN 골격 — "특성 추출 + 분류 헤드"

```text
[이미지] → (Conv + ReLU + Pool) 반복  → (Flatten) → (Linear → logits)
            특성 추출 (시각 패턴)         1D화        분류 헤드 (의사결정)
```

이 **이 분할**이 거의 모든 비전 모델의 공통 설계입니다. ViT·ResNet도 헤드만 바뀝니다.
`03_ai_vision/README.md`의 흐름도(이미지→OpenCV→인식→CNN→인지)에서 CNN에 해당.

### `ReLU`는 왜 필수인가

Conv/Linear는 **선형** 연산입니다. 선형만 쌓으면 하나의 선형으로 요약되어(합성) 결국 회귀와 같아집니다.
`ReLU(x)=max(0,x)`라는 **비선형**이 층 사이를 가로막아야 복잡한 경계를 그릴 수 있습니다.
- ReLU 외 선택지: `GELU`(Transformer·ViT), `LeakyReLU`(죽은 뉴런 보완). 기본은 ReLU.

### 05장 과적합 경계는 여기에도

합성 데이터라 100%가 쉽습니다. 실제 MNIST/자율주행 영상에서는 train 100%여도 test는 다릅니다.
**CNN은 파라미터가 많아 과적합이 더 흔**합니다 → 데이터 증강(회전·크롭), dropout, 검증 집합이
프로젝트 단계(`06_projects`)에서 필요해집니다.

## 정리

1. 이미지 = `(N, C, H, W)` tensor. OpenCV는 `(H,W,C)`, 순서 변환 필요.
2. `Conv2d`: 위치 공유 필터로 지역 패턴 추출. **파라미터가 해상도와 무관**(weight sharing).
3. `MaxPool`: 크기 축소 + 위치 강건성. `padding`/`stride`로 out shape 계산.
4. CNN = [특성 추출] + [분류 헤드]. 학습 루프 5단계는 04·05와 동일.
5. `ReLU`가 선형들의 합성을 막아 비선형 경계를 만든다.
6. 분류 손실은 `CrossEntropyLoss`(raw logits + int64 index).

## 직접 해보기

`book/code/ch06_cnn.py`:
1. `out_channels`를 4→8로 늘리면 파라미터와 수렴 속도가 어떻게 변하나요?
2. `MaxPool2d(2)`를 빼면(해상도 유지) Flatten 후 `Linear` 입력 크기를 어떻게 고쳐야 하나요?
3. 노이즈 `0.1`→`1.0`으로 키워보세요. acc가 어디서 무너지는지 보며 "검증 집합 필요"를 체감.

다음: [07장 실습 과제](07_exercises.md) — 원본 README의 과제 3개를 동작 코드로.
