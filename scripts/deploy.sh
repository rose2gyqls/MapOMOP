#!/bin/bash
# =============================================================================
# MapOMOP 배포 스크립트
# =============================================================================
# 병원 서버에서 MapOMOP 시스템을 배포합니다.
#
# 사용법:
#   ./scripts/deploy.sh [--cpu-only] [--rebuild]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 옵션 파싱
CPU_ONLY=false
REBUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --rebuild)
            REBUILD=true
            shift
            ;;
        --help)
            echo "MapOMOP 배포 스크립트"
            echo ""
            echo "사용법: ./scripts/deploy.sh [옵션]"
            echo ""
            echo "옵션:"
            echo "  --cpu-only    CPU 전용 서비스로 배포"
            echo "  --rebuild     이미지 재빌드 후 배포"
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
echo -e "${BLUE}MapOMOP 시스템 배포${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

cd "$PROJECT_DIR"

# 환경 파일 확인
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}환경 설정 파일(.env)이 없습니다. 기본값으로 생성합니다.${NC}"
    cp env.example .env
    echo -e "${GREEN}✓ .env 파일 생성 완료${NC}"
    echo -e "${YELLOW}필요한 경우 .env 파일을 수정하세요.${NC}"
    echo ""
fi

# 모델 파일 확인
echo -e "${YELLOW}모델 파일 확인 중...${NC}"

if [ ! -d "volumes/sapbert_models" ] || [ -z "$(ls -A volumes/sapbert_models 2>/dev/null)" ]; then
    echo -e "${RED}✗ SapBERT 모델이 없습니다: volumes/sapbert_models/${NC}"
    echo "  모델을 다운로드하세요."
    exit 1
fi
echo -e "${GREEN}✓ SapBERT 모델 확인됨${NC}"

if [ ! -d "volumes/llm-weights" ] || [ -z "$(ls -A volumes/llm-weights 2>/dev/null)" ]; then
    echo -e "${YELLOW}⚠ LLM 모델이 없습니다: volumes/llm-weights/${NC}"
    echo "  semantic_only 모드로 동작합니다."
fi

# 기존 컨테이너 중지
echo -e "\n${YELLOW}기존 서비스 중지 중...${NC}"
docker compose down 2>/dev/null || true
echo -e "${GREEN}✓ 기존 서비스 중지 완료${NC}"

# 이미지 재빌드 (요청 시)
if [ "$REBUILD" = true ]; then
    echo -e "\n${YELLOW}이미지 재빌드 중...${NC}"
    if [ "$CPU_ONLY" = true ]; then
        "$SCRIPT_DIR/build_images.sh" --cpu-only
    else
        "$SCRIPT_DIR/build_images.sh"
    fi
fi

# 서비스 시작
echo -e "\n${YELLOW}서비스 시작 중...${NC}"
if [ "$CPU_ONLY" = true ]; then
    docker compose --profile cpu-only up -d
    echo -e "${GREEN}✓ CPU 전용 서비스 시작됨${NC}"
else
    docker compose up -d
    echo -e "${GREEN}✓ GPU 서비스 시작됨${NC}"
fi

# 서비스 상태 확인
echo -e "\n${YELLOW}서비스 상태 확인 중...${NC}"
sleep 5

docker compose ps

# 헬스 체크
echo -e "\n${YELLOW}헬스 체크...${NC}"
sleep 10

check_service() {
    local name=$1
    local url=$2
    
    if curl -s "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $name: 정상${NC}"
        return 0
    else
        echo -e "${RED}✗ $name: 응답 없음${NC}"
        return 1
    fi
}

check_service "Elasticsearch" "http://localhost:9200"
check_service "SapBERT" "http://localhost:8000/health"
check_service "MapOMOP" "http://localhost:8080/health"

# 결과 출력
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}배포 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "서비스 엔드포인트:"
echo "  - Elasticsearch: http://localhost:9200"
echo "  - SapBERT API: http://localhost:8000"
echo "  - MapOMOP API: http://localhost:8080"
echo ""
echo "유용한 명령어:"
echo "  로그 확인: docker compose logs -f"
echo "  서비스 중지: docker compose down"
echo "  서비스 재시작: docker compose restart"

