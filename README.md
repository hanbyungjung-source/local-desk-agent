# Local Desk — Windows 데스크톱 작업 에이전트

로컬 GGUF 모델(llama.cpp 서버) 또는 외부 API를 선택해 쓰는 Windows용 채팅·작업 에이전트입니다. 화면 캡처, 파일 첨부, 작업 폴더 읽기·검색·수정, PowerShell 실행 같은 도구를 **사용자 승인 아래** 호출합니다.

[구형·이종 GPU LLM 최적화 연구](https://github.com/hanbyungjung-source/dual-gpu-llm-optimization)의 측정 환경이기도 합니다. 연구에서 채택한 배치·런타임은 이 앱의 실제 텍스트·이미지 요청으로 검증했습니다.

GitHub Copilot 에이전트와 함께 개발했으며, 요구사항·설계 결정·검증 기준은 본인이 정했습니다.

## 주요 기능

| 영역 | 내용 |
|---|---|
| 모델 | 로컬 GGUF(모델별 실행 프리셋, KV 캐시 정밀도 선택) / OpenAI 호환·Vertex 등 외부 API 프로필 |
| 도구 | 화면 캡처, 파일 첨부(PDF 등), 작업 폴더 읽기·검색·해시 기반 단일 블록 수정, 비대화형 PowerShell |
| 승인 | 도구별 `사용 안 함 / 매번 확인 / 허용`. 파일 수정·셸 실행은 변경 차이와 명령을 확인받음 |
| 코드 검색 | BM25 + Tree-sitter 기반 로컬 코드 검색·심볼 탐색(Python, C/C++, C#, Java 등) |
| 문맥 관리 | 대화 자동 압축(원문 보존), KV 캐시 재사용 표시, 호출 ID 기반 도구 이력 |
| 자원 감시 | GPU 전용·공유 메모리와 RAM 상한 감시, 초과 시 해당 서버만 중단 |
| 보안 | API 키는 Windows DPAPI로 암호화 저장, 응답에는 마스킹 표시 |

## 구성

| 경로 | 내용 |
|---|---|
| [desktop_agent/app.py](desktop_agent/app.py) | Tkinter UI |
| [desktop_agent/agent.py](desktop_agent/agent.py) | 작업 루프·도구 호출 |
| [desktop_agent/api.py](desktop_agent/api.py) | 외부 API 클라이언트(스트리밍, 재시도, 키 회전) |
| [desktop_agent/protocol.py](desktop_agent/protocol.py) | 도구 호출 스키마·검증 |
| [desktop_agent/tools.py](desktop_agent/tools.py), [workspace_tools.py](desktop_agent/workspace_tools.py), [terminals.py](desktop_agent/terminals.py) | 도구 구현 |
| [desktop_agent/retrieval.py](desktop_agent/retrieval.py) | 로컬 코드 검색 |
| [desktop_agent/residency.py](desktop_agent/residency.py) | 동적 가중치 재배치 제어 |
| [desktop_agent/benchmark_placement.py](desktop_agent/benchmark_placement.py), [benchmark_resources.py](desktop_agent/benchmark_resources.py) | GPU 배치 벤치마크·자원 감시 |
| [desktop_agent/credentials.py](desktop_agent/credentials.py) | DPAPI 기반 키 저장 |
| [desktop_agent/README.md](desktop_agent/README.md) | 기능별 상세 변경 기록 |
| [desktop_agent/TOOL-CONTRACT.md](desktop_agent/TOOL-CONTRACT.md) | 도구 입력 형식·한도·중단 정책 |
| [desktop_agent/PLACEMENT-STUDY.md](desktop_agent/PLACEMENT-STUDY.md) | GPU 배치 연구 기록 |
| [tests/](tests/) | 단위 테스트 |

모델 파일, 런타임 바이너리, 사용자 설정·세션 기록(`desktop_agent/data/`)은 포함하지 않았습니다.

## 실행

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r desktop_agent\requirements.txt
python -m playwright install chromium
python -m desktop_agent.app
```

로컬 모델을 쓰려면 llama.cpp `llama-server` 실행 파일과 GGUF 모델 경로를 설정 화면에서 지정해야 합니다.

## 테스트

```powershell
python -m unittest tests.test_desktop_agent tests.test_desktop_api
```

일부 테스트는 원 개발 환경의 런타임 배포 파일(`desktop_agent/data/`)을 전제로 하므로 이 저장소만으로는 건너뛰거나 실패할 수 있습니다.

## 한계

- PowerShell 도구의 작업 폴더는 샌드박스가 아니며, 관리자 권한·대화형 입력은 지원하지 않습니다.
- 검색 도구는 CAPTCHA·로그인 화면을 우회하지 않습니다.
- 코드 검색은 임베딩 의미 검색이나 언어 서버 수준의 참조 해석이 아닙니다.
