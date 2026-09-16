"""[책 06·07장] 초소형 CNN 이미지 분류 — torchvision 없이 합성 8x8 두 클래스.

"아래쪽 밝음" vs "위쪽 밝음" 위치 분류. 50% -> 100% 수렴을 관찰한다.
실행:  python ch06_cnn.py   (torch 2.8.0+cpu 검증)
"""
import torch

torch.manual_seed(0)


def make_toy(n_per=40):
    xs, ys = [], []
    for c in range(2):
        base = torch.zeros(n_per, 1, 8, 8)
        if c == 0:
            base[:, :, 4:8, :] = 1.0          # 아래쪽 밝음
        else:
            base[:, :, 0:4, :] = 1.0          # 위쪽 밝음
        base += 0.1 * torch.randn_like(base)  # 노이즈
        xs.append(base)
        ys.append(torch.full((n_per,), c, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


X, Y = make_toy()
perm = torch.randperm(X.size(0))              # 샘플 순서 섞음 (RNG 상태에 영향)
X, Y = X[perm], Y[perm]
net = torch.nn.Sequential(
    torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),   # (N,1,8,8)->(N,4,8,8)
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),                              # ->(N,4,4,4)
    torch.nn.Flatten(),                                 # ->(N,64)
    torch.nn.Linear(4 * 4 * 4, 2),                      # 분류 헤드
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
