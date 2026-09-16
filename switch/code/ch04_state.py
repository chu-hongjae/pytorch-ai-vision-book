"""[전향 가이드 04장] 모델 = 상태가 있는 함수 — train()/eval() 비결정성 실측.

실행: python ch04_state.py   (torch 2.8.0+cpu 검증)
"""
import torch

torch.manual_seed(0)
net = torch.nn.Sequential(
    torch.nn.Linear(4, 16), torch.nn.Dropout(0.5), torch.nn.Linear(16, 2)
)
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

net.train()                       # 학습 모드: Dropout이 뉴런을 무작위로 끈다
a1, a2 = net(x).flatten(), net(x).flatten()
print("train() 1st:", [round(v, 4) for v in a1.tolist()])
print("train() 2nd:", [round(v, 4) for v in a2.tolist()],
      "-> 다른가?", bool((a1 != a2).any()))

net.eval()                        # 추론 모드: Dropout이 꺼지고 보정된다
b1, b2 = net(x).flatten(), net(x).flatten()
print("eval()  1st:", [round(v, 4) for v in b1.tolist()])
print("eval()  2nd:", [round(v, 4) for v in b2.tolist()],
      "-> 같은가?", bool((b1 == b2).all()))

# eval()을 빼면 어떤 일이 벌어지는가: 같은 요청의 답이 흔들린다
net.train()
labels = [int(net(x).argmax()) for _ in range(200)]
print("train() 200회 label 분포:", {v: labels.count(v) for v in set(labels)})
net.eval()
with torch.no_grad():
    labels = [int(net(x).argmax()) for _ in range(200)]
print("eval()  200회 label 분포:", {v: labels.count(v) for v in set(labels)})
