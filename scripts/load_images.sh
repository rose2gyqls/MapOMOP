#!/bin/bash
# =============================================================================
# MapOMOP Docker 이미지 로드 스크립트
# =============================================================================
# 저장된 tar 파일에서 Docker 이미지를 로드합니다.
# 병원 내부망(오프라인 환경)에서 사용합니다.
#
# 사용법:
#   ./scripts/load_images.sh [--input-dir ./dist]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INPUT_DIR="${PROJECT_DIR}/dist"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 옵션 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "MapOMOP Docker 이미지 로드 스크립트"
            echo ""
            echo "사용법: ./scripts/load_images.sh [옵션]"
            echo ""
            echo "옵션:"
            echo "  --input-dir DIR  이미지 파일 디렉토리 지정 (기본값: ./dist)"
            echo "  --help           도움말 표시"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MapOMOP Docker 이미지 로드${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "입력 디렉토리: $INPUT_DIR"
echo ""

cd "$PROJECT_DIR"

# 이미지 로드 함수
load_image() {
    local file=$1
    local name=$2
    
    if [ -f "$INPUT_DIR/$file" ]; then
        echo -e "${YELLOW}로드 중: $file${NC}"
        gunzip -c "$INPUT_DIR/$file" | docker load
        echo -e "${GREEN}✓ $name 로드 완료${NC}"
    else
        echo -e "${RED}✗ $file 파일을 찾을 수 없습니다${NC}"
        return 1
    fi
}

# 이미지 로드
echo -e "${YELLOW}[1/3] SapBERT 이미지 로드 중...${NC}"
load_image "sapbert-api.tar.gz" "SapBERT"

echo -e "\n${YELLOW}[2/3] MapOMOP 이미지 로드 중...${NC}"
load_image "mapomop-api.tar.gz" "MapOMOP"

echo -e "\n${YELLOW}[3/3] Elasticsearch 이미지 로드 중...${NC}"
load_image "elasticsearch.tar.gz" "Elasticsearch"

# 결과 출력
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}로드 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "로드된 이미지:"
docker images | grep -E "mapomop|elasticsearch" | head -10
echo ""
echo "다음 단계:"
echo "  1. 환경 설정: cp env.example .env && vi .env"
echo "  2. 서비스 시작: docker compose up -d"
