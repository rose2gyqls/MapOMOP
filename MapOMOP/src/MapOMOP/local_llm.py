"""
Local sLLM Module

로컬에서 sLLM(Small Language Model)을 로드하고 사용하는 모듈입니다.
지원 모델: Qwen, TinyLlama, Hari 등

병원 내부망 (오프라인 환경)에서 사용 가능합니다.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers not installed. Local LLM features unavailable.")


class LocalLLM:
    """로컬 sLLM 추론 클래스"""
    
    # 지원 모델 목록
    SUPPORTED_MODELS = {
        "qwen": {
            "name": "Qwen/Qwen2.5-0.5B-Instruct",
            "local_path": "qwen2.5-0.5b-instruct",
            "trust_remote_code": True
        },
        "tinyllama": {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "local_path": "tinyllama-1.1b-chat",
            "trust_remote_code": False
        },
        "hari": {
            "name": "snuh/hari-q3-8b",
            "local_path": "hari-q3-8b",
            "trust_remote_code": True
        }
    }
    
    def __init__(
        self,
        model_name: str = "qwen",
        model_path: Optional[str] = None,
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        trust_remote_code: bool = True
    ):
        """
        Initialize local LLM.
        
        Args:
            model_name: 모델 이름 (qwen, tinyllama, hari) 또는 HuggingFace 모델명
            model_path: 로컬 모델 경로 (지정 시 우선 사용)
            device: 디바이스 (auto, cuda, cpu)
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 샘플링 온도
            trust_remote_code: 원격 코드 신뢰 여부
        """
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers library not installed")
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._loaded = False
        
        # Device 설정
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 모델 경로 결정
        self.model_path = self._resolve_model_path(model_name, model_path)
        self.trust_remote_code = trust_remote_code
        
        logger.info(f"LocalLLM initialized: {model_name}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Device: {self.device}")
    
    def _resolve_model_path(self, model_name: str, model_path: Optional[str]) -> str:
        """모델 경로 결정"""
        # 직접 경로가 지정된 경우
        if model_path and os.path.exists(model_path):
            return model_path
        
        # 환경변수에서 기본 경로 확인
        base_path = os.getenv("LLM_MODELS_PATH", "/app/models")
        
        # 지원 모델 목록에서 찾기
        if model_name.lower() in self.SUPPORTED_MODELS:
            model_info = self.SUPPORTED_MODELS[model_name.lower()]
            local_path = os.path.join(base_path, model_info["local_path"])
            
            if os.path.exists(local_path):
                return local_path
            
            # 로컬에 없으면 HuggingFace 모델명 반환
            return model_info["name"]
        
        # 직접 HuggingFace 모델명으로 간주
        return model_name
    
    def load(self):
        """모델 로드"""
        if self._loaded:
            return
        
        logger.info(f"Loading model: {self.model_path}")
        
        try:
            # Trust remote code 설정
            trust_remote = self.trust_remote_code
            if self.model_name.lower() in self.SUPPORTED_MODELS:
                trust_remote = self.SUPPORTED_MODELS[self.model_name.lower()]["trust_remote_code"]
            
            # Tokenizer 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=trust_remote
            )
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=trust_remote,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # Pipeline 생성
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto" if self.device == "cuda" else None
            )
            
            self._loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트
            max_new_tokens: 최대 토큰 수
            temperature: 샘플링 온도
            
        Returns:
            생성된 텍스트
        """
        if not self._loaded:
            self.load()
        
        max_tokens = max_new_tokens or self.max_new_tokens
        temp = temperature or self.temperature
        
        # 메시지 형식 구성
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Chat template 적용
            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback for models without chat template
                if system_prompt:
                    formatted_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
                else:
                    formatted_prompt = f"User: {prompt}\n\nAssistant:"
            
            # 생성
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=temp > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            generated_text = outputs[0]["generated_text"]
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        JSON 형식으로 텍스트 생성
        
        Args:
            prompt: 프롬프트 (JSON 출력 지시 포함 권장)
            system_prompt: 시스템 프롬프트
            
        Returns:
            파싱된 JSON 딕셔너리 또는 None
        """
        response = self.generate(prompt, system_prompt)
        
        try:
            # JSON 블록 추출
            text = response.strip()
            
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            
            # JSON 객체 찾기
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
            
            return json.loads(text)
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            logger.debug(f"Response was: {response}")
            return None
    
    def cleanup(self):
        """리소스 해제"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._loaded = False
        logger.info("LocalLLM resources cleaned up")
    
    @property
    def is_loaded(self) -> bool:
        """모델 로드 여부"""
        return self._loaded
    
    def __del__(self):
        """소멸자"""
        self.cleanup()


# 전역 인스턴스 (싱글톤 패턴)
_global_llm: Optional[LocalLLM] = None


def get_local_llm(
    model_name: str = "qwen",
    model_path: Optional[str] = None,
    **kwargs
) -> LocalLLM:
    """
    전역 LocalLLM 인스턴스 반환 (싱글톤)
    
    Args:
        model_name: 모델 이름
        model_path: 로컬 모델 경로
        **kwargs: LocalLLM 추가 인자
        
    Returns:
        LocalLLM 인스턴스
    """
    global _global_llm
    
    if _global_llm is None:
        _global_llm = LocalLLM(model_name=model_name, model_path=model_path, **kwargs)
    
    return _global_llm


def cleanup_global_llm():
    """전역 LLM 인스턴스 정리"""
    global _global_llm
    
    if _global_llm is not None:
        _global_llm.cleanup()
        _global_llm = None

