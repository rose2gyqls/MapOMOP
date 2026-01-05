#!/bin/bash
# =============================================================================
# MapOMOP Docker 이미지 저장 스크립트
# =============================================================================
# 빌드된 이미지를 tar 파일로 저장합니다.
# 병원 내부망(오프라인 환경)에 배포할 때 사용합니다.
#
# 사용법:
#   ./scripts/save_images.sh [--output-dir ./dist]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/dist"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 옵션 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "MapOMOP Docker 이미지 저장 스크립트"
            echo ""
            echo "사용법: ./scripts/save_images.sh [옵션]"
            echo ""
            echo "옵션:"
            echo "  --output-dir DIR  출력 디렉토리 지정 (기본값: ./dist)"
            echo "  --help            도움말 표시"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MapOMOP Docker 이미지 저장${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "출력 디렉토리: $OUTPUT_DIR"
echo ""

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"

# 모든 이미지 저장 (GPU 버전만)
echo -e "${YELLOW}[1/3] SapBERT 이미지 저장 중...${NC}"
docker save mapomop/sapbert-api:latest | gzip > "$OUTPUT_DIR/sapbert-api.tar.gz"
echo -e "${GREEN}✓ sapbert-api.tar.gz 저장 완료${NC}"

echo -e "\n${YELLOW}[2/3] MapOMOP 이미지 저장 중...${NC}"
docker save mapomop/mapomop-api:latest | gzip > "$OUTPUT_DIR/mapomop-api.tar.gz"
echo -e "${GREEN}✓ mapomop-api.tar.gz 저장 완료${NC}"

echo -e "\n${YELLOW}[3/3] Elasticsearch 이미지 저장 중...${NC}"
docker save elasticsearch:8.18.0 | gzip > "$OUTPUT_DIR/elasticsearch.tar.gz"
echo -e "${GREEN}✓ elasticsearch.tar.gz 저장 완료${NC}"

# 결과 출력
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}저장 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "생성된 파일:"
ls -lh "$OUTPUT_DIR"/*.tar.gz
echo ""
echo "총 크기: $(du -sh "$OUTPUT_DIR" | cut -f1)"
echo ""
echo "다음 단계:"
echo "  1. dist/ 폴더와 volumes/ 폴더를 병원 서버로 복사"
echo "  2. 병원 서버에서: ./scripts/load_images.sh"
echo "  3. 서비스 시작: docker compose up -d"
