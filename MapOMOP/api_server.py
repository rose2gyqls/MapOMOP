"""
MapOMOP REST API Server

병원 내부용 의료 용어 매핑 API 서버입니다.
OMOP CDM 표준 용어로 매핑하는 기능을 제공합니다.

Endpoints:
    POST /api/v1/indexing/run - 인덱싱 실행
    POST /api/v1/mapping/execute - 매핑 실행
    GET /health - 서버 상태 확인
    GET /api/v1/info - 시스템 정보
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add src path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from MapOMOP import (
    EntityMappingAPI,
    EntityInput,
    DomainID,
    MappingResult,
    ElasticsearchClient
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Server configuration from environment variables."""
    
    # Elasticsearch settings
    ES_HOST: str = os.getenv("ES_HOST", "elasticsearch")
    ES_PORT: int = int(os.getenv("ES_PORT", "9200"))
    ES_USERNAME: str = os.getenv("ES_USERNAME", "elastic")
    ES_PASSWORD: str = os.getenv("ES_PASSWORD", "snomed")
    
    # SapBERT API settings
    SAPBERT_URL: str = os.getenv("SAPBERT_URL", "http://sapbert:8000")
    
    # Local LLM settings (local sLLM usage)
    LOCAL_LLM_MODEL: str = os.getenv("LOCAL_LLM_MODEL", "qwen")  # gemma, hari, qwen
    LOCAL_LLM_PATH: str = os.getenv("LOCAL_LLM_PATH", "/app/llm-models")
    
    # Server settings
    HOST: str = os.getenv("MAPOMOP_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("MAPOMOP_PORT", "8080"))
    
    # Mapping settings
    # local_llm: 로컬 sLLM 사용, semantic_only: 임베딩만 사용, hybrid: 혼합
    SCORING_MODE: str = os.getenv("SCORING_MODE", "local_llm")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))


config = Config()


# ============================================================================
# Request/Response Models
# ============================================================================

class DataSourceType(str, Enum):
    """Data source types for indexing."""
    POSTGRES = "POSTGRES"
    CSV = "CSV"
    ATHENA = "ATHENA"


class DataSourceConfig(BaseModel):
    """Data source configuration."""
    type: DataSourceType = Field(..., description="데이터 소스 타입")
    table_name: str = Field(default="concept", description="테이블명")
    # PostgreSQL specific
    host: Optional[str] = None
    port: Optional[int] = None
    dbname: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    # CSV specific
    file_path: Optional[str] = None


class IndexingRequest(BaseModel):
    """Indexing API request."""
    target_index: str = Field(
        default="omop_concept_v1",
        description="생성할 ES 인덱스명"
    )
    data_source: DataSourceConfig = Field(..., description="데이터 소스 설정")
    batch_size: int = Field(default=100, description="배치 크기")


class IndexingResponse(BaseModel):
    """Indexing API response."""
    status: str
    message: str
    target_index: str
    records_processed: Optional[int] = None
    elapsed_time_seconds: Optional[float] = None


class MappingSource(str, Enum):
    """Mapping source options."""
    CDM = "CDM"
    ATHENA = "ATHENA"
    ES_DATA = "ES_DATA"


class MappingConfig(BaseModel):
    """Mapping configuration."""
    source: MappingSource = Field(default=MappingSource.ES_DATA, description="데이터 소스")
    use_llm: bool = Field(default=True, description="LLM 검증 사용 여부")
    llm_model: Optional[str] = Field(default=None, description="LLM 모델명")
    top_k: int = Field(default=3, description="후보군 개수")


class MappingRequest(BaseModel):
    """Mapping API request."""
    entity_name: str = Field(..., description="매핑할 엔티티명")
    domain_id: Optional[str] = Field(default=None, description="도메인 ID")
    config: MappingConfig = Field(default_factory=MappingConfig, description="매핑 설정")


class SourceEntity(BaseModel):
    """Source entity info."""
    entity_name: str
    domain_id: Optional[str]


class AlternativeConcept(BaseModel):
    """Alternative concept info."""
    concept_id: str
    concept_name: str
    vocabulary_id: str
    score: float


class MappingResponse(BaseModel):
    """Mapping API response."""
    source_entity: SourceEntity
    mapped_concept_id: str
    mapped_concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_class_id: str
    standard_concept: str
    concept_code: str
    concept_embedding: Optional[List[float]] = None
    valid_start_date: Optional[str] = None
    valid_end_date: Optional[str] = None
    invalid_reason: Optional[str] = None
    mapping_score: float
    mapping_confidence: str
    mapping_method: str
    alternative_concepts: List[AlternativeConcept] = []


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    elasticsearch: Dict[str, Any]
    sapbert: Optional[Dict[str, Any]] = None
    llm: Optional[Dict[str, Any]] = None
    version: str


class SystemInfoResponse(BaseModel):
    """System information response."""
    version: str
    scoring_mode: str
    elasticsearch_host: str
    sapbert_url: str
    local_llm_model: str
    local_llm_path: str
    confidence_threshold: float


# ============================================================================
# Global instances
# ============================================================================

es_client: Optional[ElasticsearchClient] = None
mapping_api: Optional[EntityMappingAPI] = None


# ============================================================================
# Helper functions
# ============================================================================

def get_domain_id_enum(domain_str: Optional[str]) -> Optional[DomainID]:
    """Convert domain string to DomainID enum."""
    if not domain_str:
        return None
    
    domain_map = {
        "Procedure": DomainID.PROCEDURE,
        "Condition": DomainID.CONDITION,
        "Drug": DomainID.DRUG,
        "Observation": DomainID.OBSERVATION,
        "Measurement": DomainID.MEASUREMENT,
        "Device": DomainID.DEVICE,
    }
    
    return domain_map.get(domain_str)


def mapping_result_to_response(result: MappingResult) -> MappingResponse:
    """Convert MappingResult to API response."""
    return MappingResponse(
        source_entity=SourceEntity(
            entity_name=result.source_entity.entity_name,
            domain_id=result.source_entity.domain_id.value if result.source_entity.domain_id else None
        ),
        mapped_concept_id=result.mapped_concept_id,
        mapped_concept_name=result.mapped_concept_name,
        domain_id=result.domain_id,
        vocabulary_id=result.vocabulary_id,
        concept_class_id=result.concept_class_id,
        standard_concept=result.standard_concept,
        concept_code=result.concept_code,
        concept_embedding=result.concept_embedding,
        valid_start_date=result.valid_start_date,
        valid_end_date=result.valid_end_date,
        invalid_reason=result.invalid_reason,
        mapping_score=result.mapping_score,
        mapping_confidence=result.mapping_confidence,
        mapping_method=result.mapping_method,
        alternative_concepts=[
            AlternativeConcept(
                concept_id=alt.get('concept_id', ''),
                concept_name=alt.get('concept_name', ''),
                vocabulary_id=alt.get('vocabulary_id', ''),
                score=alt.get('score', 0.0)
            )
            for alt in result.alternative_concepts
        ]
    )


async def check_service_health(url: str) -> Dict[str, Any]:
    """Check external service health."""
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url}/health")
            if response.status_code == 200:
                return {"status": "healthy", "data": response.json()}
            else:
                return {"status": "unhealthy", "code": response.status_code}
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global es_client, mapping_api
    
    # Startup
    logger.info("Starting MapOMOP API server...")
    
    # Initialize Elasticsearch client
    try:
        es_client = ElasticsearchClient(
            host=config.ES_HOST,
            port=config.ES_PORT,
            username=config.ES_USERNAME,
            password=config.ES_PASSWORD
        )
        logger.info(f"Elasticsearch connected: {config.ES_HOST}:{config.ES_PORT}")
    except Exception as e:
        logger.error(f"Elasticsearch connection failed: {e}")
        es_client = None
    
    # Initialize Mapping API
    try:
        mapping_api = EntityMappingAPI(
            es_client=es_client,
            scoring_mode=config.SCORING_MODE,
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
            local_llm_model=config.LOCAL_LLM_MODEL,
            local_llm_path=config.LOCAL_LLM_PATH
        )
        logger.info(f"EntityMappingAPI initialized (scoring_mode: {config.SCORING_MODE})")
        if config.SCORING_MODE == "local_llm":
            logger.info(f"Local LLM: {config.LOCAL_LLM_MODEL} from {config.LOCAL_LLM_PATH}")
    except Exception as e:
        logger.error(f"EntityMappingAPI initialization failed: {e}")
        mapping_api = None
    
    logger.info("MapOMOP API server ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MapOMOP API server...")
    if es_client:
        es_client.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="MapOMOP API",
    description="OMOP CDM 의료 용어 매핑 REST API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 상태 확인"""
    es_health = {}
    if es_client:
        es_health = es_client.health_check()
    else:
        es_health = {"status": "not initialized"}
    
    # Check SapBERT service
    sapbert_health = await check_service_health(config.SAPBERT_URL)
    
    # Local LLM status (로컬이므로 서비스 체크 대신 모델 상태 확인)
    llm_health = {
        "status": "local",
        "model": config.LOCAL_LLM_MODEL,
        "path": config.LOCAL_LLM_PATH,
        "mode": config.SCORING_MODE
    }
    
    overall_status = "healthy"
    if es_health.get("status") != "connected":
        overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        elasticsearch=es_health,
        sapbert=sapbert_health,
        llm=llm_health,
        version="1.0.0"
    )


@app.get("/api/v1/info", response_model=SystemInfoResponse)
async def get_system_info():
    """시스템 정보 조회"""
    return SystemInfoResponse(
        version="1.0.0",
        scoring_mode=config.SCORING_MODE,
        elasticsearch_host=f"{config.ES_HOST}:{config.ES_PORT}",
        sapbert_url=config.SAPBERT_URL,
        local_llm_model=config.LOCAL_LLM_MODEL,
        local_llm_path=config.LOCAL_LLM_PATH,
        confidence_threshold=config.CONFIDENCE_THRESHOLD
    )


@app.post("/api/v1/indexing/run", response_model=IndexingResponse)
async def run_indexing(request: IndexingRequest, background_tasks: BackgroundTasks):
    """
    인덱싱 실행
    
    선택한 소스(PostgreSQL CDM 등)로부터 표준 용어를 읽어와
    SapBERT 임베딩을 거쳐 Elasticsearch 인덱스를 생성합니다.
    """
    logger.info(f"Indexing request received: {request.target_index}")
    
    # For now, return a placeholder response
    # TODO: Implement actual indexing logic with UnifiedIndexer
    
    start_time = time.time()
    
    try:
        # Import indexing module
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'indexing'))
        
        from unified_indexer import UnifiedIndexer, create_data_source
        
        # Create data source based on request
        ds_config = request.data_source
        
        if ds_config.type == DataSourceType.POSTGRES:
            data_source = create_data_source(
                'postgres',
                host=ds_config.host or os.getenv('PG_HOST', '172.23.100.146'),
                port=ds_config.port or os.getenv('PG_PORT', '1341'),
                dbname=ds_config.dbname or os.getenv('PG_DBNAME', 'cdm_public'),
                user=ds_config.user or os.getenv('PG_USER', 'cdmreader'),
                password=ds_config.password or os.getenv('PG_PASSWORD', 'scdm2025!@')
            )
        elif ds_config.type == DataSourceType.CSV:
            data_source = create_data_source(
                'local_csv',
                data_folder=ds_config.file_path or './data/omop-cdm'
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported data source type: {ds_config.type}"
            )
        
        # Create indexer
        indexer = UnifiedIndexer(
            data_source=data_source,
            es_host=config.ES_HOST,
            es_port=config.ES_PORT,
            es_username=config.ES_USERNAME,
            es_password=config.ES_PASSWORD,
            batch_size=request.batch_size,
            include_embeddings=True
        )
        
        # Run indexing (this could be moved to background task for large datasets)
        results = indexer.index_all(
            delete_existing=True,
            tables=['concept']
        )
        
        elapsed = time.time() - start_time
        
        if results.get('concept', False):
            return IndexingResponse(
                status="success",
                message="인덱싱이 완료되었습니다.",
                target_index=request.target_index,
                elapsed_time_seconds=round(elapsed, 2)
            )
        else:
            return IndexingResponse(
                status="failed",
                message="인덱싱 중 오류가 발생했습니다.",
                target_index=request.target_index,
                elapsed_time_seconds=round(elapsed, 2)
            )
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return IndexingResponse(
            status="error",
            message=f"인덱싱 모듈 로드 실패: {str(e)}",
            target_index=request.target_index
        )
        
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        return IndexingResponse(
            status="error",
            message=f"인덱싱 오류: {str(e)}",
            target_index=request.target_index
        )


@app.post("/api/v1/mapping/execute", response_model=MappingResponse)
async def execute_mapping(request: MappingRequest):
    """
    매핑 실행
    
    입력된 텍스트 데이터를 임베딩, 검색, LLM 검증을 거쳐
    최종 OMOP CDM 표준 코드로 매핑합니다.
    """
    logger.info(f"Mapping request: {request.entity_name} (domain: {request.domain_id})")
    
    if not mapping_api:
        raise HTTPException(
            status_code=503,
            detail="Mapping API not initialized"
        )
    
    try:
        # Create entity input
        domain_id = get_domain_id_enum(request.domain_id)
        
        entity_input = EntityInput(
            entity_name=request.entity_name,
            domain_id=domain_id
        )
        
        # Execute mapping
        results = mapping_api.map_entity(entity_input)
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"No mapping found for entity: {request.entity_name}"
            )
        
        # Return best result
        best_result = max(results, key=lambda x: x.mapping_score)
        
        return mapping_result_to_response(best_result)
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Mapping error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Mapping failed: {str(e)}"
        )


@app.post("/api/v1/mapping/batch")
async def execute_batch_mapping(entities: List[MappingRequest]):
    """
    배치 매핑 실행
    
    여러 엔티티를 한 번에 매핑합니다.
    """
    if not mapping_api:
        raise HTTPException(
            status_code=503,
            detail="Mapping API not initialized"
        )
    
    results = []
    
    for entity_req in entities:
        try:
            domain_id = get_domain_id_enum(entity_req.domain_id)
            entity_input = EntityInput(
                entity_name=entity_req.entity_name,
                domain_id=domain_id
            )
            
            mapping_results = mapping_api.map_entity(entity_input)
            
            if mapping_results:
                best = max(mapping_results, key=lambda x: x.mapping_score)
                results.append({
                    "entity_name": entity_req.entity_name,
                    "status": "success",
                    "result": mapping_result_to_response(best).dict()
                })
            else:
                results.append({
                    "entity_name": entity_req.entity_name,
                    "status": "not_found",
                    "result": None
                })
                
        except Exception as e:
            results.append({
                "entity_name": entity_req.entity_name,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "total": len(entities),
        "success": sum(1 for r in results if r["status"] == "success"),
        "results": results
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api_server:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level="info"
    )

