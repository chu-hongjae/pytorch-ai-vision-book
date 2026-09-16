"""[책 03장] 학습률 실험 — 같은 데이터, lr만 바꿔 수렴/발산을 확인한다.

y = 2x + 3 데이터를 simple_classifier.py와 동일한 손 코딩 루프로 학습.
실행:  python ch03_learning_rate_lab.py   (torch 2.8.0+cpu 검증)
"""
import torch

x = torch.tensor([[1.0], [2.0], [3.0]])
y_true = torch.tensor([[5.0], [7.0], [9.0]])


def run(steps, lr, trace=()):
    W = torch.tensor([[2.0]], requires_grad=True)   # 초기값 (예제와 동일)
    b = torch.tensor([1.0], requires_grad=True)
    curve = []
    for i in range(steps):
        loss = ((x @ W + b - y_true) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            W -= lr * W.grad
            b -= lr * b.grad
            W.grad.zero_()
            b.grad.zero_()
        if (i + 1) in trace:
            curve.append((i + 1, loss.item()))
    return W.item(), b.item(), loss.item(), curve


if __name__ == "__main__":
    # lr별 최종 상태
    for lr in (0.01, 0.1, 0.2):
        W, b, loss, _ = run(10000, lr)
        print(f"lr={lr}: W={W:.4f} b={b:.4f} loss={loss:.3e}")

    # 원본 예제(simple_classifier) 지점: lr=0.01, 100스텝
    W, b, loss, curve = run(10000, 0.01, trace=(1, 10, 100, 500, 1000, 5000))
    print("\nlr=0.01 손실 곡선 (원본은 여기서 100까지):")
    for step, l in curve:
        print(f" step {step:>5}  loss {l:.6e}")

    # 안정 조건 확인 (lr<0.214)
    W, b, loss, _ = run(100, 0.2)
    print(f"\nlr=0.2 100스텝: W={W:.1f} b={b:.1f} loss={loss:.3e} (발산 진행)")
