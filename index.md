---
title: Home
layout: default
nav_order: 1
description: "Physical AI 학습 로드맵 03_ai_vision · PyTorch 기초서 + 전향 가이드"
---
# PyTorch 기초서 & AI 전향 가이드

{: .fs-5 .mb-4 }
로봇 비전 학습(`03_ai_vision`)을 위해 쓴 두 권이에요. 📕는 딥러닝이 처음인 사람을 위한 **PyTorch 기초서**,
📗는 이미 코드를 잘 쓰는 개발자를 위한 **AI 전향 가이드**입니다. 두 권 모두 하나의 공통된 약속이 있습니다. 모든 코드를 실제로 실행해서 나온 결과만 실었다는 점이죠. 추측 수치를 안 넣는 게 생각보다 어려운 약속이라, 미리 밝혀 둡니다.

---

## 📕 [PyTorch 기초서 — Tensor에서 이미지 분류까지](book/README.md)

`03_ai_vision/pytorch` 폴더의 학습 메모와 `simple_classifier.py` 한 편에서 출발한 개념서예요. 경사 하강법이라는 단 하나의 반복문을 먼저 손으로 세웁니다. 그다음 PyTorch API에게 그 일을 이월시키는 순서로 읽혀요.

| 장 | 제목 |
|----|------|
| [00](book/00_preface.md) | 서문 — PyTorch가 뭐예요? |
| [01](book/01_tensors.md) | Tensor — 데이터를 담는 그릇 |
| [02](book/02_autograd.md) | Autograd — 기울기의 자동화 |
| [03](book/03_loss_optimizer.md) | Loss와 Optimizer — 학습률로 수렴과 발산을 경험 |
| [04](book/04_first_model.md) | 첫 모델 — `nn.Module`로 다시 쓰기 |
| [05](book/05_training_loop.md) | 실제 학습 루프 — 미니배치·DataLoader·과적합 |
| [06](book/06_cnn_vision.md) | 이미지 분류 — CNN의 기본 구조 |
| [07](book/07_exercises.md) | 실습 과제와 해답 |
| [부록 A](book/appendix_cheatsheet.md) | API 치트시트 · 용어 · 트러블슈팅 |

## 📗 [FE/BE 개발자를 위한 AI 전향 가이드](switch/README.md)

이미 코드를 잘 쓰는 사람과, 코드 짜는 방식으로 AI를 시작하려는 사람의 간극을 메우는 책이에요. "로직을 쓴다 → 로직을 데이터에서 피팅한다"는 패러다임 전환을 실측으로 체감하게 해줍니다. 기존 개발 스킬이 그대로 먹히는 지점도 있고, 역으로 그 스킬이 함정이 되는 지점도 있으니까요.

| 장 | 제목 |
|----|------|
| [00](switch/00_preface.md) | 전향자를 위한 서문 — 무엇이 정말 다른가 |
| [01](switch/01_paradigm.md) | 규칙을 '쓴다' vs '배운다' — 결정 경계 실험 |
| [02](switch/02_skill_mapping.md) | 아는 것으로 시작하기 — 개발↔AI 개념 매핑 |
| [03](switch/03_data.md) | 데이터가 곧 코드다 |
| [04](switch/04_model.md) | 모델 = 상태가 있는 함수 |
| [05](switch/05_eval.md) | 지표의 함정 — 직관이 꺾이는 세 지점 |
| [06](switch/06_serving.md) | 드디어 출격 — 추론을 API로 |
| [07](switch/07_first_project.md) | 첫 프로젝트와 90일 계획 |
| [부록 B](switch/appendix_glossary.md) | 용어 대조표 · 생태계 · FAQ |

---

## 검증 원칙

- **환경**: PyTorch 2.8.0+cpu (CPU 전용, GPU 불필요). `torch`·`numpy`·`sklearn` 범위 안에서만 다룹니다.
- **출처**: 모든 코드 블록과 수치는 이 환경에서 실행한 stdout입니다. 추측 수치는 쓰지 않았습니다.
- **재현**: 난수 예제는 `manual_seed`로 고정했습니다. 각 장의 코드는 `*/code/`에 독립 실행 파일로 있습니다.

## 이 둘의 관계

개념이 처음이면 📕PyTorch 기초서를 먼저 펼치세요. 이미 개발 경력이 있고 방향만 AI로 돌리는 중이라면 📗전향 가이드가 먼저여도 됩니다. 전향 가이드는 📕의 API 사용을 전제하지 않거든요.

[Physical AI 전체 로드맵](../README.md) · [03_ai_vision 단계](../README.md)
