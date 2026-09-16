"""[책 04장] simple_classifier.py를 nn.Module으로 — 원본과 같은 문제의 API 전환판.

y = 2x + 3 를 nn.Linear + MSELoss + SGD로 학습 (lr=0.1).
실행:  python ch04_first_model.py   (torch 2.8.0+cpu 검증)
"""
import torch

torch.manual_seed(0)
X = torch.tensor([[1.0], [2.0], [3.0]])
Y = torch.tensor([[5.0], [7.0], [9.0]])

model = torch.nn.Linear(1, 1)                 # 가중치 1·bias 1 자동 생성
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(200):                       # 학습 루프 5단계
    optimizer.zero_grad()                      # ①
    out = model(X)                             # ② forward
    loss = criterion(out, Y)                   # ③ 손실
    loss.backward()                            # ④ 역전파
    optimizer.step()                           # ⑤ 갱신
    if epoch in (0, 49, 199):
        print(f"epoch {epoch:>3}  loss {loss.item():.6f}")

print("W =", round(model.weight.item(), 4), " b =", round(model.bias.item(), 4))
print("pred(10) =", round(model(torch.tensor([[10.0]])).item(), 3), "(expect 23)")
