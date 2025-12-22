#!/bin/bash
# =============================================================================
# MapOMOP Docker 이미지 빌드 스크립트
# =============================================================================
# 모든 서비스의 Docker 이미지를 빌드합니다.
#
# 사용법:
#   ./scripts/build_images.sh [--cpu-only] [--no-cache]
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
NO_CACHE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MapOMOP Docker 이미지 빌드${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

cd "$PROJECT_DIR"

# SapBERT 이미지 빌드
echo -e "${YELLOW}[1/4] SapBERT API 이미지 빌드 중...${NC}"
if [ "$CPU_ONLY" = true ]; then
    docker build $NO_CACHE -t mapomop/sapbert-api:cpu -f sapbert-api/Dockerfile.cpu sapbert-api/
    echo -e "${GREEN}✓ mapomop/sapbert-api:cpu 빌드 완료${NC}"
else
    docker build $NO_CACHE -t mapomop/sapbert-api:latest -f sapbert-api/Dockerfile sapbert-api/
    docker build $NO_CACHE -t mapomop/sapbert-api:cpu -f sapbert-api/Dockerfile.cpu sapbert-api/
    echo -e "${GREEN}✓ mapomop/sapbert-api:latest 빌드 완료${NC}"
    echo -e "${GREEN}✓ mapomop/sapbert-api:cpu 빌드 완료${NC}"
fi

# MapOMOP API 이미지 빌드
echo -e "\n${YELLOW}[2/4] MapOMOP API 이미지 빌드 중...${NC}"
if [ "$CPU_ONLY" = true ]; then
    docker build $NO_CACHE -t mapomop/mapomop-api:cpu -f MapOMOP/Dockerfile.cpu MapOMOP/
    echo -e "${GREEN}✓ mapomop/mapomop-api:cpu 빌드 완료${NC}"
else
    docker build $NO_CACHE -t mapomop/mapomop-api:latest -f MapOMOP/Dockerfile MapOMOP/
    docker build $NO_CACHE -t mapomop/mapomop-api:cpu -f MapOMOP/Dockerfile.cpu MapOMOP/
    echo -e "${GREEN}✓ mapomop/mapomop-api:latest 빌드 완료${NC}"
    echo -e "${GREEN}✓ mapomop/mapomop-api:cpu 빌드 완료${NC}"
fi

# Elasticsearch 이미지 풀
echo -e "\n${YELLOW}[3/4] Elasticsearch 이미지 풀...${NC}"
docker pull elasticsearch:8.18.0
echo -e "${GREEN}✓ elasticsearch:8.18.0 풀 완료${NC}"

# Kibana 이미지 풀
echo -e "\n${YELLOW}[4/4] Kibana 이미지 풀...${NC}"
docker pull kibana:8.18.0
echo -e "${GREEN}✓ kibana:8.18.0 풀 완료${NC}"

# 결과 출력
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}빌드 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "빌드된 이미지:"
docker images | grep -E "mapomop|elasticsearch|kibana" | head -10
echo ""
echo "다음 단계:"
echo "  1. 이미지 저장: ./scripts/save_images.sh"
echo "  2. 서비스 시작: docker compose up -d"

