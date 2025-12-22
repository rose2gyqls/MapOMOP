# MapOMOP 시스템 배포 가이드

## 개요

이 문서는 MapOMOP 시스템을 병원 내부망(오프라인 환경)에 배포하기 위한 상세 가이드입니다.

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Hospital Network                         │
│  ┌─────────────┐      ┌──────────────────────────────────┐ │
│  │   Client    │      │         GPU Server               │ │
│  │   (PC)      │ HTTP │  ┌────────────────────────────┐  │ │
│  │             │──────│  │    Docker Container        │  │ │
│  │  - Browser  │      │  │  ┌──────────┐ ┌─────────┐  │  │ │
│  │  - Python   │      │  │  │ MapOMOP  │ │SapBERT  │  │  │ │
│  │    Client   │      │  │  │ :8080    │ │:8000    │  │  │ │
│  │             │      │  │  └────┬─────┘ └────┬────┘  │  │ │
│  └─────────────┘      │  │       │            │       │  │ │
│                       │  │  ┌────┴────────────┴────┐  │  │ │
│                       │  │  │   Elasticsearch      │  │  │ │
│                       │  │  │   :9200              │  │  │ │
│                       │  │  └──────────────────────┘  │  │ │
│                       │  └────────────────────────────┘  │ │
│                       └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 사전 요구사항

### 개발 환경 (인터넷 연결 필요)
- macOS 또는 Linux
- Docker Desktop 20.10+
- 충분한 디스크 공간 (최소 50GB)

### 배포 환경 (병원 서버)
- Linux (Ubuntu 20.04+ 권장)
- Docker 20.10+
- Docker Compose 2.0+
- (GPU 사용 시) NVIDIA Docker
- 최소 메모리: 16GB
- 최소 디스크: 100GB

## 빠른 시작

### 개발 환경에서 패키지 생성

```bash
# 1. 프로젝트 디렉토리로 이동
cd MapOMOP_Project

# 2. 스크립트 실행 권한 부여
chmod +x scripts/*.sh

# 3. 전체 패키지 생성 (GPU 버전)
./scripts/package_all.sh

# 또는 CPU 전용 패키지 생성
./scripts/package_all.sh --cpu-only
```

### 병원 서버에서 배포

```bash
# 1. 패키지 압축 해제
tar -xzf mapomop-package-YYYYMMDD.tar.gz
cd mapomop

# 2. 이미지 로드
./scripts/load_images.sh

# 3. 환경 설정
cp env.example .env
vi .env  # 병원 환경에 맞게 수정

# 4. 서비스 시작
docker compose up -d  # GPU 버전
# 또는
docker compose --profile cpu-only up -d  # CPU 버전
```

## 상세 배포 가이드

### 1. Docker 이미지 빌드

```bash
./scripts/build_images.sh
```

빌드되는 이미지:
- `mapomop/sapbert-api:latest` - SapBERT GPU 버전
- `mapomop/sapbert-api:cpu` - SapBERT CPU 버전
- `mapomop/mapomop-api:latest` - MapOMOP GPU 버전
- `mapomop/mapomop-api:cpu` - MapOMOP CPU 버전

### 2. 이미지 저장

```bash
./scripts/save_images.sh
```

생성되는 파일:
- `dist/sapbert-api-gpu.tar.gz`
- `dist/sapbert-api-cpu.tar.gz`
- `dist/mapomop-api-gpu.tar.gz`
- `dist/mapomop-api-cpu.tar.gz`
- `dist/elasticsearch.tar.gz`

### 3. 모델 파일 준비

#### SapBERT 모델
이미 `volumes/sapbert_models/`에 다운로드되어 있습니다.

#### LLM 모델
`volumes/llm-weights/` 폴더에 다음 모델 중 하나 이상:
- `qwen2.5-0.5b-instruct/`
- `tinyllama-1.1b-chat/`
- `hari-q3-8b/`

### 4. 환경 변수 설정

`env.example`을 `.env`로 복사하고 필요한 값을 수정합니다:

```bash
cp env.example .env
```

주요 설정:
```bash
# Elasticsearch
ES_PASSWORD=your-password

# 포트 설정
ES_PORT=9200
SAPBERT_PORT=8000
MAPOMOP_PORT=8080

# 스코어링 모드
SCORING_MODE=local_llm  # or semantic_only

# LLM 모델
LOCAL_LLM_MODEL=qwen  # qwen, tinyllama, hari
```

## 서비스 관리

### 서비스 시작/중지

```bash
# GPU 버전 시작
docker compose up -d

# CPU 버전 시작
docker compose --profile cpu-only up -d

# 서비스 중지
docker compose down

# 서비스 재시작
docker compose restart
```

### 로그 확인

```bash
# 전체 로그
docker compose logs -f

# 특정 서비스 로그
docker compose logs -f mapomop
docker compose logs -f sapbert
docker compose logs -f elasticsearch
```

### 상태 확인

```bash
# 컨테이너 상태
docker compose ps

# 헬스 체크
curl http://localhost:8080/health
curl http://localhost:8000/health
curl http://localhost:9200/_cluster/health
```

## API 사용 예시

### 매핑 요청

```bash
curl -X POST http://localhost:8080/api/v1/mapping/execute \
  -H "Content-Type: application/json" \
  -d '{
    "entity_name": "Acetaminophen",
    "domain_id": "Drug",
    "config": {
      "use_llm": true,
      "top_k": 5
    }
  }'
```

### 인덱싱 요청

```bash
curl -X POST http://localhost:8080/api/v1/indexing/run \
  -H "Content-Type: application/json" \
  -d '{
    "target_index": "omop_concepts",
    "data_source": {
      "type": "POSTGRES",
      "table_name": "concept",
      "host": "your-postgres-host",
      "port": 5432,
      "dbname": "cdm_db",
      "user": "user",
      "password": "password"
    },
    "tables": ["concept", "synonym"]
  }'
```

## 문제 해결

### Elasticsearch 시작 실패

```bash
# vm.max_map_count 확인 및 수정
sudo sysctl -w vm.max_map_count=262144

# 영구 설정
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
```

### GPU 인식 안됨

```bash
# NVIDIA Docker 설치 확인
nvidia-docker --version

# GPU 상태 확인
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### 메모리 부족

`.env` 파일에서 메모리 설정 조정:
```bash
ES_JAVA_OPTS=-Xms1g -Xmx1g
```

## 파일 구조

```
MapOMOP_Project/
├── docker-compose.yml      # 통합 Compose 파일
├── env.example             # 환경 변수 템플릿
├── DEPLOYMENT.md           # 배포 가이드 (이 문서)
├── MapOMOP/                # MapOMOP API
│   ├── Dockerfile
│   ├── Dockerfile.cpu
│   ├── api_server.py
│   └── ...
├── sapbert-api/            # SapBERT API
│   ├── Dockerfile
│   ├── Dockerfile.cpu
│   ├── app.py
│   └── ...
├── scripts/                # 배포 스크립트
│   ├── build_images.sh
│   ├── save_images.sh
│   ├── load_images.sh
│   ├── deploy.sh
│   └── package_all.sh
└── volumes/                # 모델 파일
    ├── sapbert_models/
    └── llm-weights/
```

## 버전 정보

- MapOMOP: 1.0.0
- SapBERT API: 1.0.0
- Elasticsearch: 8.18.0

