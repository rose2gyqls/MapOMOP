#!/bin/bash
# =============================================================================
# MapOMOP Docker 이미지 로드 스크립트
# =============================================================================
# 저장된 tar 파일에서 Docker 이미지를 로드합니다.
# 병원 내부망(오프라인 환경)에서 사용합니다.
#
# 사용법:
#   ./scripts/load_images.sh [--cpu-only] [--input-dir ./dist]
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
CPU_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --input-dir)
            INPUT_DIR="$2"
            shift 2
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
        docker load < "$INPUT_DIR/$file"
        echo -e "${GREEN}✓ $name 로드 완료${NC}"
    else
        echo -e "${RED}✗ $file 파일을 찾을 수 없습니다${NC}"
        return 1
    fi
}

if [ "$CPU_ONLY" = true ]; then
    # CPU 전용 이미지만 로드
    load_image "sapbert-api-cpu.tar.gz" "SapBERT CPU"
    load_image "mapomop-api-cpu.tar.gz" "MapOMOP CPU"
    load_image "elasticsearch.tar.gz" "Elasticsearch"
else
    # GPU 이미지 로드 (있는 경우)
    if [ -f "$INPUT_DIR/sapbert-api-gpu.tar.gz" ]; then
        load_image "sapbert-api-gpu.tar.gz" "SapBERT GPU"
    fi
    load_image "sapbert-api-cpu.tar.gz" "SapBERT CPU"
    
    if [ -f "$INPUT_DIR/mapomop-api-gpu.tar.gz" ]; then
        load_image "mapomop-api-gpu.tar.gz" "MapOMOP GPU"
    fi
    load_image "mapomop-api-cpu.tar.gz" "MapOMOP CPU"
    
    load_image "elasticsearch.tar.gz" "Elasticsearch"
fi

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
echo "  2. GPU 서비스 시작: docker compose up -d"
echo "  3. CPU 서비스 시작: docker compose --profile cpu-only up -d"

