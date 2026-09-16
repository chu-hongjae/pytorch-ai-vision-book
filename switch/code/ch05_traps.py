"""[전향 가이드 05장·03장] 지표의 함정 3종 — 정확도 역설 / 과적합 / 타깃 누수.

실행: python ch05_traps.py   (torch 2.8.0+cpu / sklearn / numpy 검증)
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)

print("=" * 64)
print("함정 1  정확도의 역설 (불균형)")
print("=" * 64)
X = np.vstack([rng.normal(-0.4, 1.2, (1900, 3)),      # 다수 클래스
               rng.normal(1.2, 1.2, (100, 3))])        # 소수 클래스(위험)
y = np.r_[np.zeros(1900, int), np.ones(100, int)]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
mdl = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
pred = mdl.predict(Xte)
print(f"positives in test: {int(yte.sum())} / {len(yte)} ({yte.mean():.1%})")
print("model accuracy         =", round(accuracy_score(yte, pred), 4))
print("trivial 'always 0' acc =", round(accuracy_score(yte, np.zeros_like(yte)), 4))
print("balanced accuracy      =", round(balanced_accuracy_score(yte, pred), 4))
print("confusion matrix (rows=true 0/1):")
print(confusion_matrix(yte, pred))
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
print(f"minority recall = {tp / (tp + fn):.4f}   precision = {tp / (tp + fp):.4f}")

print()
print("=" * 64)
print("함정 2  과적합: 용량을 늘리면 train만 좋아진다")
print("=" * 64)
torch.manual_seed(0)
n = 20
xs = torch.linspace(0, 2 * np.pi, n).reshape(-1, 1)
ys = torch.sin(xs) + 0.4 * torch.randn(n, 1)
xtr, xva, ytr, yva = train_test_split(xs.numpy(), ys.numpy(),
                                      test_size=0.5, random_state=0)
xtr, ytr, xva, yva = map(torch.tensor, (xtr, ytr, xva, yva))


def fit(hidden, wd, epochs=1500):
    m = torch.nn.Sequential(torch.nn.Linear(1, hidden), torch.nn.ReLU(),
                            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                            torch.nn.Linear(hidden, 1))
    o = torch.optim.Adam(m.parameters(), lr=0.03, weight_decay=wd)
    for _ in range(epochs):
        o.zero_grad()
        torch.nn.functional.mse_loss(m(xtr), ytr).backward()
        o.step()
    m.eval()
    with torch.no_grad():
        return (torch.nn.functional.mse_loss(m(xtr), ytr).item(),
                torch.nn.functional.mse_loss(m(xva), yva).item())


for hidden, wd, tag in ((2, 0.0, "과소적합"), (32, 0.0, "과적합"), (32, 0.02, "과대규제")):
    tr, va = fit(hidden, wd)
    print(f"hidden={hidden:>2} wd={wd:<5} [{tag}]  train MSE={tr:.5f}  val MSE={va:.5f}")

print()
print("=" * 64)
print("함정 3  target leakage: 정답이 입력에 스며들면 검증이 무의미해진다")
print("=" * 64)
N = 3000
churn = (rng.random(N) < 0.2).astype(int)              # 20% 이탈
usage = rng.normal(0, 1, N) - churn * 1.2              # 이탈과 상관 있는 정상 피처
# leak: '해지 문의'는 예측 시점에 아직 존재하지 않는, 라벨 이후 정보
cancel_calls = np.where(churn == 1, rng.integers(2, 6, N),
                        rng.integers(0, 2, N)).astype(float)
feat_good = np.column_stack([usage, rng.normal(0, 1, N), rng.normal(0, 1, N)])
feat_leak = np.column_stack([usage, cancel_calls])

for name, F in (("정상(사용량만)", feat_good), ("누수(+해지문의)", feat_leak)):
    Xtr_, Xte_, ytr_, yte_ = train_test_split(F, churn, test_size=0.3,
                                              random_state=0, stratify=churn)
    m2 = LogisticRegression(max_iter=1000).fit(Xtr_, ytr_)
    baseline = max(np.bincount(yte_)) / len(yte_)       # 항상-다수클래스 상수함수
    print(f"{name:<16} train acc={accuracy_score(ytr_, m2.predict(Xtr_)):.4f}"
          f"  test acc={accuracy_score(yte_, m2.predict(Xte_)):.4f}"
          f"  (항상-기준선 {baseline:.4f})")
print("cancel_calls는 '이미 해지를 결심한 고객'에게만 존재 → 서비스 시점엔 값이 없음")
