# SapBERT API Server

의료 엔티티 텍스트를 벡터로 변환하는 SapBERT 임베딩 API 서버입니다.

## 개요

SapBERT(Self-Alignment Pretrained BERT)는 의료/생물학 도메인에 특화된 언어 모델로, 
의료 용어의 의미적 유사성을 효과적으로 캡처합니다.

이 API 서버는 MapOMOP 시스템의 임베딩 서비스로 사용되며, 
텍스트를 768차원 벡터로 변환합니다.

## API 엔드포인트

### POST /embed

텍스트를 벡터로 변환합니다.

**Request:**
```json
{
  "text": "myocardial ischemia",
  "normalize": true
}
```

또는 배치 처리:
```json
{
  "text": ["myocardial ischemia", "hypertension", "diabetes"],
  "normalize": true
}
```

**Response:**
```json
{
  "embeddings": [[0.123, -0.456, ...]],
  "dimension": 768,
  "count": 1,
  "processing_time_ms": 45.32
}
```

### GET /health

서버 상태를 확인합니다.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "gpu_available": true,
  "gpu_name": "NVIDIA A100"
}
```

### GET /info

모델 정보를 반환합니다.

**Response:**
```json
{
  "model_name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
  "embedding_dimension": 768,
  "max_length": 25,
  "batch_size": 128,
  "device": "cuda:0"
}
```

## 실행 방법

### Docker Compose (권장)

```bash
# GPU 버전
docker compose up -d sapbert

# CPU 버전
docker compose --profile cpu-only up -d sapbert-cpu
```

### Docker Build

```bash
# GPU 버전
docker build -t sapbert-api:latest .

# CPU 버전
docker build -f Dockerfile.cpu -t sapbert-api:cpu .

# 실행
docker run -d -p 8000:8000 --gpus all sapbert-api:latest
```

### 로컬 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 실행
python app.py

# 또는
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `SAPBERT_MODEL_NAME` | HuggingFace 모델명 | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` |
| `SAPBERT_MODEL_PATH` | 로컬 모델 경로 (선택) | `""` |
| `SAPBERT_MAX_LENGTH` | 최대 토큰 길이 | `25` |
| `SAPBERT_BATCH_SIZE` | 배치 크기 | `128` |
| `SAPBERT_DEVICE` | 디바이스 (auto/cuda/cpu) | `auto` |
| `SAPBERT_HOST` | 서버 호스트 | `0.0.0.0` |
| `SAPBERT_PORT` | 서버 포트 | `8000` |

## 오프라인 환경 배포 (병원 내부망)

이 프로젝트는 이미 SapBERT 모델이 `volumes/sapbert_models/`에 다운로드되어 있습니다.

### 현재 모델 파일 구조

```
volumes/sapbert_models/
├── model.safetensors    # 모델 가중치 (418MB)
├── config.json          # 모델 설정
├── tokenizer.json       # 토크나이저
├── tokenizer_config.json
├── vocab.txt
└── special_tokens_map.json
```

### Docker Compose로 실행 (권장)

```bash
cd sapbert-api

# GPU 버전
docker compose up -d sapbert

# CPU 버전
docker compose --profile cpu-only up -d sapbert-cpu
```

모델은 자동으로 `../volumes/sapbert_models`에서 마운트됩니다.

### 수동 Docker 실행

```bash
# GPU 버전
docker run -d -p 8000:8000 \
  -v $(pwd)/../volumes/sapbert_models:/app/models:ro \
  -e SAPBERT_MODEL_PATH=/app/models \
  --gpus all \
  sapbert-api:latest

# CPU 버전
docker run -d -p 8000:8000 \
  -v $(pwd)/../volumes/sapbert_models:/app/models:ro \
  -e SAPBERT_MODEL_PATH=/app/models \
  sapbert-api:cpu
```

### 병원 서버 배포용 이미지 저장

```bash
# 이미지 빌드 및 tar 저장
./scripts/build_and_save.sh

# 생성된 파일:
#   dist/sapbert-api-gpu.tar.gz
#   dist/sapbert-api-cpu.tar.gz
```

### 병원 서버에서 로드

```bash
# 이미지 로드
docker load < sapbert-api-gpu.tar.gz

# 모델 파일과 함께 실행
docker run -d -p 8000:8000 \
  -v /path/to/sapbert_models:/app/models:ro \
  -e SAPBERT_MODEL_PATH=/app/models \
  --gpus all \
  sapbert-api:latest
```

### 모델 직접 다운로드 (필요시)

모델이 없는 경우에만:
```bash
python scripts/download_model.py --output-dir ../volumes/sapbert_models --verify
```

## 성능 참고

- **GPU (A100)**: ~500 텍스트/초
- **GPU (RTX 3090)**: ~300 텍스트/초
- **CPU (16 cores)**: ~30 텍스트/초

배치 크기를 조절하여 메모리 사용량과 처리 속도를 최적화할 수 있습니다.

## 테스트

```bash
# Health check
curl http://localhost:8000/health

# 단일 텍스트 임베딩
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "myocardial ischemia"}'

# 배치 임베딩
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": ["hypertension", "diabetes", "asthma"]}'
```

## 라이선스

MIT License

