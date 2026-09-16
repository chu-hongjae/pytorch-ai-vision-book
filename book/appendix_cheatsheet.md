---
title: "부록 A · 치트시트 · 트러블슈팅"
layout: default
nav_order: 11
parent: "PyTorch 기초서"
description: "부록 A · API 치트시트 · 용어 · 트러블슈팅"
---

# 부록 A · API 치트시트 · 용어 · 트러블슈팅

> 이 책 01~07장에서 실제로 쓴 API만 모았습니다. 챕터에서 돌아가며 한 번씩 찾는 용어집.

처음부터 끝까지 읽으라고 만든 파일은 아닙니다. 코드 중간에 "이거 뭐라고 썼더라" 하고 펼치는 자리예요.
표는 그대로 두었고, 표와 표 사이에만 말을 조금 얹었습니다.

## A.1 tensor 생성·연산 (01장)

손이 가장 자주 가는 것들입니다. 오른쪽 열은 외우려 하지 마시고, 필요할 때 이 표로 돌아오세요.

| 작업 | 코드 |
|------|------|
| 만들기 | `torch.tensor([...])`, `torch.zeros(2,3)`, `torch.randn(2,3)` |
| 난수 재현 | `torch.manual_seed(0)` |
| 모양/타입/장치 | `t.shape`, `t.dtype`, `t.device` |
| 개수/합/평균 | `t.numel()`, `t.sum()`, `t.mean()` |
| 스칼라 추출 | `t.item()` (원 1개일 때만) |
| 리스트 변환 | `t.tolist()` |
| 원소별 곱 | `a * b` |
| 행렬곱 | `a @ b`  (= `torch.matmul`) |
| 모양 변경 | `a.reshape(r,c)`, `a.T`, `a.squeeze()`, `a.unsqueeze(0)` |
| 타입 변경 | `a.float()`, `a.long()` |
| 장치 이동 | `a.to('cuda')` (GPU 있을 때) |

## A.2 autograd (02장)

주의 열은 단순한 참고가 아니라 경고입니다. 특히 기울기가 누적되는 문제는 안 걸려본 사람이 오히려
드물어요.

| 작업 | 코드 | 주의 |
|------|------|------|
| 기울기 추적 켜기 | `t.requires_grad_(True)` / 생성 시 `requires_grad=True` | leaf에만 `.grad` |
| 역전파 | `loss.backward()` | `loss`는 스칼라여야 |
| 기울기 읽기 | `t.grad` | 누적됨 → `zero_grad()` |
| 추적 끄기 | `with torch.no_grad():` | 갱신·추론 |

## A.3 모델·손실·optimizer (03~06장)

복붙용 블록입니다. 이 책에서 실제로 조합해서 쓴 형태만 모아뒀어요.

```python
# 모델
m  = torch.nn.Linear(784, 10)               # 단일 층
m  = torch.nn.Sequential(
        torch.nn.Conv2d(1,16,3,padding=1),  # (C_in,C_out,k)
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(..., 10),
     )
ps = m.parameters()                          # 학습 대상 모음

# 손실
torch.nn.MSELoss()(pred, target)             # 회귀
torch.nn.CrossEntropyLoss()(logits, target)  # 분류 (logits raw, target int64)

# optimizer
opt = torch.optim.SGD(m.parameters(), lr=0.1)
opt = torch.optim.Adam(m.parameters(), lr=0.01)   # 기본 강력

# 데이터
ds  = torch.utils.data.TensorDataset(X, Y)
dl  = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)

# 저장/추론
torch.save(m.state_dict(), "m.pt")
m.load_state_dict(torch.load("m.pt")); m.eval()
```

## A.4 학습 루프 5단계 — 이 책 전체의 압축

이 책의 결론은 전부 아래 블록 안에 있습니다. 다른 장을 다 잊어도 이 순서만 기억하면 다시 조립할 수
있어요.

```python
for epoch in range(N):
    for xb, yb in dl:
        opt.zero_grad()      # ① 기울기 초기화
        out  = model(xb)     # ② forward
        loss = criterion(out, yb)  # ③ 손실
        loss.backward()      # ④ 역전파
        opt.step()           # ⑤ 파라미터 갱신
```

## A.5 용어집

정의가 길지 않습니다. 한 줄 사전이라고 생각하시고, 헷갈리는 단어만 꺼내 보세요.

| 용어 | 뜻 |
|------|----|
| tensor | 같은 타입 숫자의 격자 + 연산/autograd |
| leaf | 사용자가 만든, `.grad`가 채워지는 tensor |
| forward / backward | 예측 계산 / 기울기 역전파 |
| loss / criterion | 틀린 정도(스칼라) / 손실 함수 |
| gradient (`grad`) | loss의 파라미터 미분값 ( 갱신 방향) |
| learning rate (lr) | 경사 하강 걸음 크기 |
| epoch / step | 전체 데이터 1순회 / batch 1처리 |
| batch / mini-batch | 한 스텝에 쓰는 샘플 묶음 |
| logits | softmax 이전의 raw 점수 |
| overfitting | train만 맞고 unseen은 틀림 |
| broadcasting | size-1 축 자동 복제 |
| state_dict | 파라미터 이름→tensor 사전 (저장 단위) |
| `nn.functional.*` | 층 객체 없이 함수만 쓰는 API |

## A.6 트러블슈팅 — 흔한 에러와 원인

증상 열에서 눈이 멈추면 그 줄만 보시면 됩니다. 에러 메시지는 실제 출력에서 옮겨온 문자열이라, 그대로
복사해서 검색창에 쳐도 답이 나와요.

| 증상 | 1순위 원인 | 확인 |
|------|-----------|------|
| 학습 안 됨(loss 정체) | 에포크 부족 / lr 과소 | 03장 곡선, lr↑ |
| loss가 `nan`/발산 | lr 과대 / `zero_grad` 누락 | 03장 안정조건, ① 확인 |
| `mat1 and mat2 shapes cannot be multiplied` | 선형대수 shape 불일치 | 축 계산표(01장 1.2) |
| `expected scalar type Float but found Long` / dtype 에러 | `int64`를 float에 곱함 | `.float()` |
| `target should contain at least class idx` / CrossEntropy shape | one-hot/float 레이블 | `target.long()`, (N,) |
| `a leaf Variable that requires grad is being used in an in-place operation` | 추적 tensor를 직접 수정 | `with no_grad()` |
| GPU 코드 CPU에서 에러 | device 불일치 | `.to('cpu')` |
| 추론 결과가 매번 다름 | `eval()`/`no_grad()` 누락 | 5.5 |
| `torch.load` 보안 경고 / 실패 | pickle 로드 | `weights_only=True` 권장 |

## A.7 확인해야 할 다음 개념 (04 이후 단계)

이 책은 "학습 루프 이해"에서 멈췄습니다. 아쉬우시겠지만, 그게 이 책의 역할이었어요. 실전으로 가려면
순서대로 다음을 보시면 됩니다.

1. **검증·평가**: train/val/test split, `accuracy`, confusion matrix, early stopping.
2. **규제**: dropout, weight decay, 데이터 증강 (06장 과적합 대비).
3. **전이 학습**: 사전학습 ResNet/ViT 헤드만 교체 (`torchvision.models`).
4. **로봇 연결**: 카메라 입력 파이프라인(`opencv`) → 추론 → 제어. `03_ai_vision` 흐름도의 끝.
5. **성능**: GPU(`.to('cuda')`), mixed precision, 서빙(ONNX/TorchScript).

## A.8 추천 자료 (원본 README의 확장)

바로 펼쳐볼 수 있는 것만 걸었습니다. 링크는 원본에 있던 것을 그대로 두고, 옆에 용도 칸을 붙였어요.

| 자료 | 링크 | 용도 |
|------|------|------|
| PyTorch 공식 튜토리얼 | https://pytorch.org/tutorials/ | 장별 공식 예제 |
| PyTorch Examples | https://github.com/pytorch/examples | MNIST/CNN 실코드 |
| PyTorch API 문서 | https://pytorch.org/docs/stable/ | 시그니처 확인 |
| Hugging Face Transformers | https://github.com/huggingface/transformers | 이후 NLP/ViT |
| 3Blue1Brown Neural Nets | https://www.3blue1brown.com/topics/neural-networks | 직관 영상 |
| fast.ai Practical DL | https://course.fast.ai/ | 실전 중심 |

> 원본 `README.md`의 추천 목록(pytorch, pytorch/examples, transformers)을 유지하며,
> 튜토리얼·문서·직관 자료를 보강했습니다.
