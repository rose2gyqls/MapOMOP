#!/bin/bash
# SapBERT API 이미지 빌드 및 저장 스크립트
# 오프라인 환경 배포용

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/dist"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SapBERT API 이미지 빌드 및 저장${NC}"
echo -e "${GREEN}========================================${NC}"

cd "$PROJECT_DIR"

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# GPU 버전 빌드
echo -e "\n${YELLOW}[1/4] GPU 버전 이미지 빌드 중...${NC}"
docker build -t sapbert-api:latest -t sapbert-api:gpu .

# CPU 버전 빌드
echo -e "\n${YELLOW}[2/4] CPU 버전 이미지 빌드 중...${NC}"
docker build -f Dockerfile.cpu -t sapbert-api:cpu .

# GPU 버전 저장
echo -e "\n${YELLOW}[3/4] GPU 버전 이미지 저장 중...${NC}"
docker save sapbert-api:latest | gzip > "$OUTPUT_DIR/sapbert-api-gpu.tar.gz"
echo -e "${GREEN}✓ 저장 완료: $OUTPUT_DIR/sapbert-api-gpu.tar.gz${NC}"

# CPU 버전 저장
echo -e "\n${YELLOW}[4/4] CPU 버전 이미지 저장 중...${NC}"
docker save sapbert-api:cpu | gzip > "$OUTPUT_DIR/sapbert-api-cpu.tar.gz"
echo -e "${GREEN}✓ 저장 완료: $OUTPUT_DIR/sapbert-api-cpu.tar.gz${NC}"

# 결과 출력
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}빌드 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "생성된 파일:"
ls -lh "$OUTPUT_DIR"/*.tar.gz
echo ""
echo "병원 서버에서 로드하려면:"
echo "  docker load < sapbert-api-gpu.tar.gz"
echo "  docker load < sapbert-api-cpu.tar.gz"

