#!/bin/bash
# =============================================================================
# MapOMOP Docker 이미지 빌드 스크립트
# =============================================================================
# 모든 서비스의 Docker 이미지를 빌드합니다.
# Mac에서 linux/amd64 플랫폼으로 GPU 이미지를 빌드합니다.
#
# 사용법:
#   ./scripts/build_images.sh [--no-cache]
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
NO_CACHE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --help)
            echo "MapOMOP Docker 이미지 빌드 스크립트"
            echo ""
            echo "사용법: ./scripts/build_images.sh [옵션]"
            echo ""
            echo "옵션:"
            echo "  --no-cache  캐시 없이 빌드"
            echo "  --help      도움말 표시"
            exit 0
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
echo "플랫폼: linux/amd64"
echo ""

cd "$PROJECT_DIR"

# SapBERT 이미지 빌드 (GPU)
echo -e "${YELLOW}[1/3] SapBERT API 이미지 빌드 중...${NC}"
docker build $NO_CACHE --platform linux/amd64 -t mapomop/sapbert-api:latest -f sapbert-api/Dockerfile sapbert-api/
echo -e "${GREEN}✓ mapomop/sapbert-api:latest 빌드 완료${NC}"

# MapOMOP API 이미지 빌드 (GPU)
echo -e "\n${YELLOW}[2/3] MapOMOP API 이미지 빌드 중...${NC}"
docker build $NO_CACHE --platform linux/amd64 -t mapomop/mapomop-api:latest -f MapOMOP/Dockerfile MapOMOP/
echo -e "${GREEN}✓ mapomop/mapomop-api:latest 빌드 완료${NC}"

# Elasticsearch 이미지 풀
echo -e "\n${YELLOW}[3/3] Elasticsearch 이미지 풀...${NC}"
docker pull --platform linux/amd64 elasticsearch:8.18.0
echo -e "${GREEN}✓ elasticsearch:8.18.0 풀 완료${NC}"

# 결과 출력
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}빌드 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "빌드된 이미지:"
docker images | grep -E "mapomop|elasticsearch" | head -10
echo ""
echo "다음 단계:"
echo "  1. 이미지 저장: ./scripts/save_images.sh"
echo "  2. 서비스 시작: docker compose up -d"
