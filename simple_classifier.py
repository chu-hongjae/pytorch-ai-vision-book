import torch

# Simple linear model: y = W*x + b
W = torch.tensor([[2.0]], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

x = torch.tensor([[1.0], [2.0], [3.0]])
y_true = torch.tensor([[5.0], [7.0], [9.0]])

for _ in range(100):
    y_pred = x @ W + b
    loss = ((y_pred - y_true) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        W -= 0.01 * W.grad
        b -= 0.01 * b.grad
        W.grad.zero_()
        b.grad.zero_()

print(f"W = {W.item():.4f}")
print(f"b = {b.item():.4f}")
