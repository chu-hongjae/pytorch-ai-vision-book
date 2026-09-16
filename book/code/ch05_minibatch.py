"""[책 05장] 미니배치 학습 루프 — DataLoader + 5단계 batch 반복.

노이즈 있는 y = 2x + 3 회귀(N=100)를 batch=16으로 학습.
실행:  python ch05_minibatch.py   (torch 2.8.0+cpu 검증)
"""
import torch

torch.manual_seed(0)
N = 100
X = torch.rand(N, 1) * 10                     # x ∈ [0,10)
Y = 2 * X + 3 + torch.randn(N, 1) * 0.5       # y = 2x+3 + 노이즈

model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
ds = torch.utils.data.TensorDataset(X, Y)
dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)

for epoch in range(300):
    total = 0.0
    for xb, yb in dl:                         # batch마다 5단계
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)     # 개수 가중 평균
    if epoch in (0, 99, 299):
        print(f"epoch {epoch:>3}  avg train loss {total / N:.4f}")

print("W =", round(model.weight.item(), 3), " b =", round(model.bias.item(), 3))
