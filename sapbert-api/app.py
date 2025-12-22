"""
SapBERT API Server

REST API server for generating SapBERT embeddings for medical entity names.
Provides HTTP endpoints for text-to-vector conversion.

Endpoints:
    POST /embed - Generate embeddings for text(s)
    POST /embed/batch - Batch embedding generation
    GET /health - Health check
    GET /info - Model information
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer

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
    
    # Model settings
    # 로컬 모델 경로 우선 사용 (병원 내부망 오프라인 환경용)
    MODEL_PATH: str = os.getenv(
        "SAPBERT_MODEL_PATH", 
        "/app/models"  # Docker 컨테이너 내부 마운트 경로
    )
    MODEL_NAME: str = os.getenv(
        "SAPBERT_MODEL_NAME", 
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    )
    
    # Processing settings
    MAX_LENGTH: int = int(os.getenv("SAPBERT_MAX_LENGTH", "25"))
    BATCH_SIZE: int = int(os.getenv("SAPBERT_BATCH_SIZE", "128"))
    
    # Device settings
    DEVICE: str = os.getenv("SAPBERT_DEVICE", "auto")  # auto, cuda, cpu
    
    # Server settings
    HOST: str = os.getenv("SAPBERT_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("SAPBERT_PORT", "8000"))


config = Config()


# ============================================================================
# Request/Response Models
# ============================================================================

class EmbedRequest(BaseModel):
    """Single or batch embedding request."""
    text: Union[str, List[str]] = Field(
        ..., 
        description="Text or list of texts to embed"
    )
    normalize: bool = Field(
        default=True, 
        description="Whether to L2-normalize embeddings"
    )


class EmbedResponse(BaseModel):
    """Embedding response."""
    embeddings: List[List[float]] = Field(
        ..., 
        description="List of embedding vectors"
    )
    dimension: int = Field(
        ..., 
        description="Embedding dimension"
    )
    count: int = Field(
        ..., 
        description="Number of embeddings generated"
    )
    processing_time_ms: float = Field(
        ..., 
        description="Processing time in milliseconds"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    gpu_available: bool
    gpu_name: Optional[str] = None


class InfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    embedding_dimension: int
    max_length: int
    batch_size: int
    device: str


# ============================================================================
# SapBERT Embedder
# ============================================================================

class SapBERTEmbedder:
    """SapBERT embedding service."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_name = config.MODEL_NAME
        self.max_length = config.MAX_LENGTH
        self.batch_size = config.BATCH_SIZE
        self._loaded = False
    
    def load(self):
        """Load the SapBERT model."""
        if self._loaded:
            return
        
        logger.info(f"Loading SapBERT model: {self.model_name}")
        start_time = time.time()
        
        # Determine device
        if config.DEVICE == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.DEVICE)
        
        logger.info(f"Using device: {self.device}")
        
        # 로컬 모델 경로 우선 사용 (오프라인 환경 지원)
        # 1. SAPBERT_MODEL_PATH 환경변수로 지정된 경로
        # 2. 경로가 없거나 유효하지 않으면 HuggingFace에서 다운로드
        model_path = config.MODEL_PATH
        
        if model_path and os.path.exists(model_path):
            # 로컬 경로에 모델 파일이 있는지 확인
            required_files = ["config.json", "tokenizer_config.json"]
            has_model_files = all(
                os.path.exists(os.path.join(model_path, f)) for f in required_files
            )
            if has_model_files:
                logger.info(f"Loading model from local path: {model_path}")
            else:
                logger.warning(f"Model files not found in {model_path}, falling back to HuggingFace")
                model_path = self.model_name
        else:
            logger.info(f"Local path not available, loading from HuggingFace: {self.model_name}")
            model_path = self.model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            self._loaded = True
            
            elapsed = time.time() - start_time
            logger.info(f"Model loaded successfully in {elapsed:.2f}s")
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self.model is None:
            return 768  # Default for BERT-based models
        return self.model.config.hidden_size
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    def encode(
        self, 
        texts: List[str], 
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            Numpy array of embeddings
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        
        if not texts:
            return np.array([])
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = self._encode_batch(batch_texts)
                all_embeddings.append(batch_embeddings)
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        if normalize:
            # L2 normalization
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            embeddings = embeddings / norms
        
        return embeddings
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts."""
        # Tokenize
        tokens = self.tokenizer.batch_encode_plus(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        # Get model output
        outputs = self.model(**tokens)
        
        # Use CLS token embedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        return cls_embeddings.cpu().numpy()
    
    def cleanup(self):
        """Release resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._loaded = False
        logger.info("Resources cleaned up")


# Global embedder instance
embedder = SapBERTEmbedder()


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting SapBERT API server...")
    embedder.load()
    logger.info("SapBERT API server ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SapBERT API server...")
    embedder.cleanup()
    logger.info("Shutdown complete")


app = FastAPI(
    title="SapBERT API",
    description="REST API for SapBERT medical entity embeddings",
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
    """Health check endpoint."""
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    
    return HealthResponse(
        status="healthy" if embedder.is_loaded else "degraded",
        model_loaded=embedder.is_loaded,
        device=str(embedder.device) if embedder.device else "not initialized",
        gpu_available=torch.cuda.is_available(),
        gpu_name=gpu_name
    )


@app.get("/info", response_model=InfoResponse)
async def model_info():
    """Get model information."""
    if not embedder.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return InfoResponse(
        model_name=embedder.model_name,
        embedding_dimension=embedder.embedding_dim,
        max_length=embedder.max_length,
        batch_size=embedder.batch_size,
        device=str(embedder.device)
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """
    Generate embeddings for text(s).
    
    - **text**: Single text string or list of texts
    - **normalize**: Whether to L2-normalize embeddings (default: True)
    
    Returns embedding vectors for the input text(s).
    """
    if not embedder.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    # Handle single text or list
    if isinstance(request.text, str):
        texts = [request.text]
    else:
        texts = request.text
    
    if not texts:
        raise HTTPException(status_code=400, detail="Empty text input")
    
    if len(texts) > 1000:
        raise HTTPException(
            status_code=400, 
            detail="Too many texts. Maximum 1000 texts per request."
        )
    
    try:
        embeddings = embedder.encode(texts, normalize=request.normalize)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return EmbedResponse(
            embeddings=embeddings.tolist(),
            dimension=embedder.embedding_dim,
            count=len(embeddings),
            processing_time_ms=round(elapsed_ms, 2)
        )
        
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/batch", response_model=EmbedResponse)
async def embed_batch(texts: List[str], normalize: bool = True):
    """
    Batch embedding endpoint (alternative format).
    
    Accepts a list of texts directly in the request body.
    """
    request = EmbedRequest(text=texts, normalize=normalize)
    return await embed(request)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level="info"
    )

