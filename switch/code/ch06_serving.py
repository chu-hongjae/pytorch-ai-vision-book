"""[전향 가이드 06장] 추론을 HTTP API로 — stdlib http.server + httpx 지연 실측.

실행: python ch06_serving.py   (torch 2.8.0+cpu / httpx 검증)
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from sklearn.metrics import accuracy_score

rng = np.random.default_rng(0)

# 01장 모델을 다시 학습해 서빙한다
torch.manual_seed(0)
X = torch.from_numpy(rng.uniform(-2, 2, (600, 2)).astype(np.float32))
Y = ((X ** 2).sum(1) < 1.0).long()
net = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
opt = torch.optim.Adam(net.parameters(), lr=0.01)
for _ in range(300):
    opt.zero_grad()
    torch.nn.functional.cross_entropy(net(X), Y).backward()
    opt.step()
net.eval()
print("served model circle test acc =",
      round(accuracy_score(Y.numpy(), net(X).argmax(1).numpy()), 4))

# 가중치만 저장 → 아키텍처는 코드로 따로 복원 (06.1의 두 산출물)
PATH = "model.pt"
torch.save(net.state_dict(), PATH)


class Handler(BaseHTTPRequestHandler):
    model = None                                   # 기동 시 1회 주입 (06.2)

    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404)
            return
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with torch.no_grad():
            p = torch.softmax(self.model(torch.tensor([b["features"]],
                                                       dtype=torch.float32)), 1).flatten()
        raw = json.dumps({"label": int(p.argmax()),
                          "confidence": round(float(p.max()), 4)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):                     # 로그 소음 제거
        pass


Handler.model = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.ReLU(),
                                    torch.nn.Linear(32, 2))
Handler.model.load_state_dict(torch.load(PATH, weights_only=True))
Handler.model.eval()

srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{port}/predict"

import httpx

# (1) 순수 추론 (HTTP 제외)
x0 = torch.tensor([[0.1, 0.1]])
with torch.no_grad():
    Handler.model(x0)                              # 웜업
    t0 = time.perf_counter()
    for _ in range(1000):
        Handler.model(x0)
    pure = (time.perf_counter() - t0) / 1000 * 1000

# (2) 클라이언트를 재사용하는 정석
client = httpx.Client(timeout=10)
for _ in range(20):
    client.post(url, json={"features": [0.5, 0.5]})
t0 = time.perf_counter()
for _ in range(200):
    client.post(url, json={"features": [0.5, 0.5]})
http_ms = (time.perf_counter() - t0) / 200 * 1000

# (3) 매 요청 새 클라이언트 — 흔한 실수
t0 = time.perf_counter()
for _ in range(200):
    httpx.post(url, json={"features": [0.5, 0.5]})
cold_ms = (time.perf_counter() - t0) / 200 * 1000

print("POST [0,0]     ->", client.post(url, json={"features": [0.0, 0.0]}).json())
print("POST [1.9,1.9] ->", client.post(url, json={"features": [1.9, 1.9]}).json())
print(f"pure inference   = {pure:.3f} ms/req")
print(f"HTTP warm client = {http_ms:.2f} ms/req")
print(f"HTTP new-client  = {cold_ms:.2f} ms/req  <- 클라이언트 생성 오버헤드")

client.close()
srv.shutdown()
os.remove(PATH)
