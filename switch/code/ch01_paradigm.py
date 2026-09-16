"""[전향 가이드 01장] 규칙을 '쓴다' vs '배운다' — 원형 결정 경계 경쟁 실험.

실행: python ch01_paradigm.py   (torch 2.8.0+cpu / numpy / sklearn 검증)
"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(4000, 2)).astype(np.float32)
y = ((X ** 2).sum(1) < 1.0).astype(np.int64)          # 반지름 1 원 안 = 1 (정답은 숨김)

print("positive ratio:", y.mean())                     # 항상-0 상수함수의 한계

# ── 라운드 1: 사람이 축 임계값을 손으로 쓴다 ─────────────────────
best, best_rule = 0.0, None
for axis in range(2):
    for thr in np.linspace(-2, 2, 801):
        for sign in (1, -1):
            acc = np.mean(((sign * (X[:, axis] > thr)).astype(np.int64) == y))
            if acc > best:
                best, best_rule = acc, (axis, float(thr), sign)
print("A1 단순 임계값 최선 acc =", round(best, 4), "rule axis/thr/sign =", best_rule)

# ── 라운드 2: 사람이 형태를 맞혀 다이아몬드를 쓴다 ────────────────
best_d = 0.0
for c in np.linspace(0.1, 4, 800):
    acc = np.mean(((np.abs(X[:, 0]) + np.abs(X[:, 1]) < c).astype(np.int64) == y))
    best_d = max(best_d, acc)
print("A2 |x|+|y|<c 다이아몬드 최선 acc =", round(best_d, 4))

# ── 라운드 3: 형태를 주지 않고 데이터로 피팅한다 ──────────────────
torch.manual_seed(0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
net = torch.nn.Sequential(
    torch.nn.Linear(2, 32), torch.nn.ReLU(),
    torch.nn.Linear(32, 32), torch.nn.ReLU(),
    torch.nn.Linear(32, 2),
)
opt = torch.optim.Adam(net.parameters(), lr=0.01)
lossf = torch.nn.CrossEntropyLoss()
Xtr_t, ytr_t, Xte_t, yte_t = map(torch.tensor, (Xtr, ytr, Xte, yte))
for _ in range(400):
    opt.zero_grad()
    loss = lossf(net(Xtr_t), ytr_t)
    loss.backward()
    opt.step()
net.eval()
with torch.no_grad():
    tr = accuracy_score(ytr, net(Xtr_t).argmax(1).numpy())
    te = accuracy_score(yte, net(Xte_t).argmax(1).numpy())
print(f"A3 MLP train acc = {tr:.4f}  test acc = {te:.4f}")
print("   last loss =", round(loss.item(), 6))

# ── 보조: 피처만 잘 만들면 모델이 필요 없다 (03장 A0) ─────────────
r2 = (X ** 2).sum(1)
print("A0 피처 x^2+y^2 + 임계값 1.0 acc =", accuracy_score(y, (r2 < 1.0).astype(np.int64)))
