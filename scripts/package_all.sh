#!/bin/bash
# =============================================================================
# MapOMOP 전체 패키지 생성 스크립트
# =============================================================================
# Docker 이미지, 모델 파일, 설정 파일을 하나의 tar 파일로 패키징합니다.
# 병원 내부망 배포용 완전한 패키지를 생성합니다.
# Mac에서 linux/amd64 플랫폼으로 GPU 이미지를 빌드합니다.
#
# 사용법:
#   ./scripts/package_all.sh [--output mapomop-package.tar.gz] [--no-models]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TEMP_DIR=$(mktemp -d)
OUTPUT_FILE="mapomop-package-$(date +%Y%m%d).tar.gz"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 옵션 파싱
INCLUDE_MODELS=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --no-models)
            INCLUDE_MODELS=false
            shift
            ;;
        --help)
            echo "MapOMOP 전체 패키지 생성 스크립트"
            echo ""
            echo "사용법: ./scripts/package_all.sh [옵션]"
            echo ""
            echo "옵션:"
            echo "  --no-models   모델 파일 제외"
            echo "  --output FILE 출력 파일명 지정"
            echo "  --help        도움말 표시"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MapOMOP 전체 패키지 생성${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "플랫폼: linux/amd64"
echo "출력 파일: $OUTPUT_FILE"
echo "임시 디렉토리: $TEMP_DIR"
echo ""

cd "$PROJECT_DIR"

# 클린업 함수
cleanup() {
    echo -e "\n${YELLOW}임시 파일 정리 중...${NC}"
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# 패키지 디렉토리 구조 생성
mkdir -p "$TEMP_DIR/mapomop/images"
mkdir -p "$TEMP_DIR/mapomop/volumes/sapbert_models"
mkdir -p "$TEMP_DIR/mapomop/volumes/llm-weights"
mkdir -p "$TEMP_DIR/mapomop/scripts"

# 1. Docker 이미지 빌드
echo -e "${YELLOW}[1/5] Docker 이미지 빌드 중...${NC}"
"$SCRIPT_DIR/build_images.sh"
echo -e "${GREEN}✓ 이미지 빌드 완료${NC}"

# 2. Docker 이미지 저장
echo -e "\n${YELLOW}[2/5] Docker 이미지 저장 중...${NC}"
docker save mapomop/sapbert-api:latest | gzip > "$TEMP_DIR/mapomop/images/sapbert-api.tar.gz"
docker save mapomop/mapomop-api:latest | gzip > "$TEMP_DIR/mapomop/images/mapomop-api.tar.gz"
docker save elasticsearch:8.18.0 | gzip > "$TEMP_DIR/mapomop/images/elasticsearch.tar.gz"
echo -e "${GREEN}✓ 이미지 저장 완료${NC}"

# 3. 모델 파일 복사
if [ "$INCLUDE_MODELS" = true ]; then
    echo -e "\n${YELLOW}[3/5] 모델 파일 복사 중...${NC}"
    
    # SapBERT 모델
    if [ -d "volumes/sapbert_models" ] && [ "$(ls -A volumes/sapbert_models 2>/dev/null)" ]; then
        echo "  SapBERT 모델 복사 중..."
        cp -r volumes/sapbert_models/* "$TEMP_DIR/mapomop/volumes/sapbert_models/"
        echo -e "  ${GREEN}✓ SapBERT 모델 복사 완료${NC}"
    else
        echo -e "  ${RED}✗ SapBERT 모델을 찾을 수 없습니다${NC}"
    fi
    
    # LLM 모델
    if [ -d "volumes/llm-weights" ] && [ "$(ls -A volumes/llm-weights 2>/dev/null)" ]; then
        echo "  LLM 모델 복사 중..."
        # 다운로드 스크립트 제외
        find volumes/llm-weights -mindepth 1 -maxdepth 1 -type d -exec cp -r {} "$TEMP_DIR/mapomop/volumes/llm-weights/" \;
        echo -e "  ${GREEN}✓ LLM 모델 복사 완료${NC}"
    else
        echo -e "  ${YELLOW}⚠ LLM 모델이 없습니다${NC}"
    fi
else
    echo -e "\n${YELLOW}[3/5] 모델 파일 건너뜀 (--no-models 옵션)${NC}"
fi

# 4. 설정 파일 복사
echo -e "\n${YELLOW}[4/5] 설정 파일 복사 중...${NC}"
cp docker-compose.yml "$TEMP_DIR/mapomop/"
cp env.example "$TEMP_DIR/mapomop/"
cp -r scripts/*.sh "$TEMP_DIR/mapomop/scripts/"
chmod +x "$TEMP_DIR/mapomop/scripts/"*.sh
echo -e "${GREEN}✓ 설정 파일 복사 완료${NC}"

# 설치 가이드 생성
cat > "$TEMP_DIR/mapomop/INSTALL.md" << 'EOF'
# MapOMOP 설치 가이드

## 사전 요구사항
- Docker 20.10 이상
- Docker Compose 2.0 이상
- NVIDIA Docker (GPU 사용)

## 설치 단계

### 1. 이미지 로드
```bash
cd mapomop
./scripts/load_images.sh
```

### 2. 환경 설정
```bash
cp env.example .env
# 필요한 설정 수정
vi .env
```

### 3. 서비스 시작
```bash
docker compose up -d
```

### 4. 상태 확인
```bash
docker compose ps
curl http://localhost:8080/health
```

## 서비스 엔드포인트
- MapOMOP API: http://localhost:8080
- SapBERT API: http://localhost:8000
- Elasticsearch: http://localhost:9200

## 문제 해결
```bash
# 로그 확인
docker compose logs -f

# 서비스 재시작
docker compose restart
```
EOF

echo -e "${GREEN}✓ 설치 가이드 생성 완료${NC}"

# 5. 최종 패키징
echo -e "\n${YELLOW}[5/5] 최종 패키지 생성 중...${NC}"
cd "$TEMP_DIR"
tar -czf "$PROJECT_DIR/$OUTPUT_FILE" mapomop/

# 결과 출력
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}패키지 생성 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "생성된 파일: $PROJECT_DIR/$OUTPUT_FILE"
ls -lh "$PROJECT_DIR/$OUTPUT_FILE"
echo ""
echo "패키지 내용:"
tar -tzf "$PROJECT_DIR/$OUTPUT_FILE" | head -20
echo "... (총 $(tar -tzf "$PROJECT_DIR/$OUTPUT_FILE" | wc -l) 파일)"
echo ""
echo "배포 방법:"
echo "  1. $OUTPUT_FILE 파일을 병원 서버로 복사"
echo "  2. tar -xzf $OUTPUT_FILE"
echo "  3. cd mapomop && ./scripts/load_images.sh"
echo "  4. docker compose up -d"
