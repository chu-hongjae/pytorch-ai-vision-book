---
title: "06 · 드디어 출격 — 추론을 API로"
layout: default
nav_order: 27
parent: "FE/BE → AI 전향 가이드"
description: "state_dict → HTTP /predict. 병목은 모델이 아니라 HTTP 사용법이었다"
---
# 06 · 드디어 출격 — 추론을 API로

> 목적: 노트북 밖으로 나온다. 그리고 **병목이 모델이 아님을 숫자로 확인한다.**
> 선행: 03·05장. 소요: 35분. 코드: `code/ch06_serving.py` (실제 HTTP 왕복 측정 포함)

여기까지 와서 "아직 서비스는 안 돼 보여요"라고들 합니다. 아닙니다. **추론은 그냥 HTTP API입니다.**
당신이 이미 하는 일이고, 이 책 00장이 promised한 "이곳에서 당신이 이긴다"가 실현되는 장입니다.

`fastapi`·`uvicorn`이 이 머신에 없어 실망하지 마세요. 아래처럼 **표준 라이브러리 `http.server`**
로도 같은 결론이 나옵니다. 프레임워크는 껍데기이고 병목은 그 바깥에 있습니다.

## 6.1 배포 산출물은 2개다

04장에서 본 대로, 배포 대상은 코드가 하나 아닙니다.

| 산출물 | 내용 | 저장 단위 | 버전 관리 |
|--------|------|-----------|-----------|
| 아키텍처 | `nn.Module` 클래스 정의 | 코드 | git |
| 가중치 | `state_dict` + 전처리 상수(평균/분산/인코딩 사전) | 파일(수 MB~) | 별도(예: `model_v17.pt`) |

**이 둘이 짝이 맞아야 합니다.** shape은 맞는데 의미가 다른 값을 얹으면 에러 없이 틀린 답이 나옵니다.

```python
import torch

PATH = "model.pt"
torch.save({"state_dict": net.state_dict(), "input_dim": 2, "version": "v1"}, PATH)

blob = torch.load(PATH, weights_only=True)     # pickle 전체 객체 저장은 금지 권장
net2 = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.ReLU(),
                           torch.nn.Linear(32, 2))   # 아키텍처를 정확히 다시 구성
net2.load_state_dict(blob["state_dict"])
net2.eval()                                     # 04장: 결정화
print("loaded:", blob["version"], blob["input_dim"])
```

- `state_dict`만 저장: 이식성 최고, 역직렬화 코드 실행 위험 없음.
- `weights_only=True`: 로드 시 임의 코드 실행을 막습니다. **서드파티 파일을 받을 때 필수.**
- `eval()`: 04장 실측대로 이게 빠지면 **같은 입력에 다른 답**이 나옵니다.

## 6.2 서빙은 상태가 있는 프로세스다

가장 흔한 배포 사고 3개, 전부 당신이 아는 패턴입니다.

| 사고 | 일반 개발에서의 이름 | 해법 |
|------|---------------------|------|
| 요청마다 모델을 load | 매 요청 DB 커넥션 생성 | **기동 시 1회 load** (싱글턴/전역) |
| 워커마다 다른 가중치 | 캐시 불일치 | 이미지/레플리카에 같은 버전 번들 |
| 추론 중 `optimizer.step()` | 읽기 전용인데 쓰임 | 학습·추론 프로세스 분리(04.4) |

`Handler.model`처럼 전역에 한 번 올려두는 게 정답입니다. 아래 실측이 그 근거입니다.

## 6.3 실측 — 순수 추론 vs HTTP 왕복 vs 나쁜 클라이언트

`/predict` 엔드포인트를 만들어 01장 원형 분류 모델을 올려놓고 세 가지를 재봤습니다.

```python
import httpx, time, torch

# (1) HTTP를 totally 제외하고 모델.forward 만 1000회
x0 = torch.tensor([[0.1, 0.1]])
with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(1000):
        Handler.model(x0)
    pure = (time.perf_counter() - t0) / 1000 * 1000

# (2) 재사용 클라이언트로 200회 왕복 (웜업 후)
client = httpx.Client(timeout=10)
t0 = time.perf_counter()
for _ in range(200):
    client.post(url, json={"features": [0.5, 0.5]})
http_ms = (time.perf_counter() - t0) / 200 * 1000

# (3) 매 요청 새 클라이언트 생성 (흔한 실수)
t0 = time.perf_counter()
for _ in range(200):
    httpx.post(url, json={"features": [0.5, 0.5]})
cold_ms = (time.perf_counter() - t0) / 200 * 1000
```

```text
pure inference   = 0.027 ms/req
HTTP warm client = 1.96 ms/req
HTTP new-client  = 324.69 ms/req  <- 클라이언트 생성 오버헤드
```

**결론을 크게 쓰겠습니다. 병목은 모델이 아니라 클라이언트 사용법입니다.**

- 모델 계산: 0.027 ms
- HTTP 왕복(연결 재사용): 1.96 ms → 모델의 72배, **그래도 2ms 미만**
- 클라이언트를 매번 생성: 324.69 ms → 모델의 **12,000배**

"추론이 느려요"라는 이야기의 상당수가 (3)번입니다. 이걸 잡는 데 ML 지식은 0이 필요하고
**HTTP 커넥션 풀 상식**만 필요합니다. 이 책 00장 0.3표가 여기서 payoff됩니다.

### 응답 contract도 당신의 영역

```text
POST features=[0,0]    -> {'label': 1, 'confidence': 1.0}
POST features=[1.9,1.9] -> {'label': 0, 'confidence': 1.0}
```

`label`만 주지 말고 `confidence`를 함께 주세요. **확신이 낮은 요청을 사람이 처리하도록 라우팅**하는
최소 장치가 그 하나입니다. 그리고 이 값이 distribution drift를 감지하는 유일한 무료 신호입니다
(→ 05장 5.4 체크리스트의 "모니터링").

## 6.4 batch와 캐시 — 시스템 설계로 이기는 곳

| 전략 | 언제 | 효과 |
|------|------|------|
| **응답 캐시** (입력 hash → 결과) | 입력 중복이 많을 때 | 가장 싸고 효과 큼. ML을 아예 안 태움 |
| **배치 추론** | 오프라인·동시 다요청 | GPU/스레드 효율 급상승. 단 지연 ↑ |
| **Dynamic batching** | 초당 요청 많 + 지연 허용 | 서버 측에서 10ms 모아서 한 번에. 구현 비용 높음 |
| **증류/양자화/Prune** | 지연 급한 엣지·로봇 | 모델 크기·속도 절감. 정확도 트레이드오프 |
| **온디바이스 vs 서버** | 로봇·실시간 제어 | 센서 루프 안에서 추론하면 네트워크 지연이 사라짐 |

**마지막 줄이 Physical AI에서 결정적입니다.** 로봇 제어 루프가 50Hz(20ms 주기)라면,
서버 왕복 2ms도 쫄깃하지만 네트워크 지터는 치명적입니다. 그래서 이 저장소 상위 로드맵이
"시뮬레이션 검증 → 실제 하드웨어" 순서를 강제하는 것과 이어집니다.
(참고: [Physical AI 학습 요약](../../../physical-ai-study-summary.md) 의 5장·10장)

## 6.5 최소 배포 스크립트 (그대로 도는 형태)

`code/ch06_serving.py`는 다음을 하나의 파일로 합친 것입니다:
학습 → 저장 → 로드 → `ThreadingHTTPServer` 구동 → `httpx` 왕복 측정.
`fastapi` 없이도 결론이 같다는 걸 보이려는 의도입니다.

```python
class Handler(BaseHTTPRequestHandler):
    model = None                                   # 기동 시 1회 주입 (6.2)

    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404); return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with torch.no_grad():
            probs = torch.softmax(self.model(
                torch.tensor([body["features"]], dtype=torch.float32)), 1).flatten()
        raw = json.dumps({"label": int(probs.argmax()),
                          "confidence": round(float(probs.max()), 4)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
```

**이 코드에서 ML인 부분은 한 줄**입니다(`self.model(...)`). 나머지는 전부 백엔드입니다.

## 정리

1. 배포 산출물은 **아키텍처(코드) + 가중치·전처리 상수(파일)** 두 개이고 짝 맞춰 올려야 한다.
2. 서빙 프로세스는 상태(가중치)를 가진다. 기동 시 1회 load, 워커 간 동일 버전, 갱신 금지.
3. 실측: 추론 0.027ms, 왕복 1.96ms, **클라이언트 재생성 324.69ms.** 느리다고 느끼는 건 대부분 셋째.
4. `confidence`를 응답에 넣어 사람 개입 라우팅과 드리프트 감지의 씨앗으로 쓴다.
5. batch·캐시·온디바이스 결정은 ML이 아니라 **시스템 설계** 문제고, 그건 당신의 주무대다.

## 직접 해보기

`code/ch06_serving.py`:
1. `Handler.model.eval()`을 빼고 200번 왕복해서 `label`이 흔들리는지 보세요(04장 + 06장 결합 사고).
2. 입력 length 검증을 추가하세요. `features`가 2개가 아니면 400을 반환하게 — **당신이 이미 하는 일**입니다.
3. 6.4의 응답 캐시를 붙이고命中率을 측정해 보세요. 이 문제에선 몇 %가 히트하나요?

다음: [07. 첫 프로젝트와 90일 계획](07_first_project.md)
