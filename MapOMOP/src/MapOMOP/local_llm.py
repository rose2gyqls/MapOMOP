"""
Local sLLM Module

Module for loading and using Small Language Models (sLLM) locally.
Supported models: Gemma, Hari, Qwen

Can be used in offline environments (e.g., hospital internal networks).
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
    """Local sLLM inference class"""
    
    # Supported models list
    SUPPORTED_MODELS = {
        "gemma": {
            "name": "google/gemma-2-2b",
            "local_path": "gemma-2-2b",
            "trust_remote_code": False
        },
        "hari": {
            "name": "snuh/hari-q3-8b",
            "local_path": "hari-q3-8b",
            "trust_remote_code": True
        },
        "qwen": {
            "name": "Qwen/Qwen2.5-3B",
            "local_path": "qwen2.5-3b",
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
            model_name: Model name (gemma, hari, qwen) or HuggingFace model identifier
            model_path: Local model path (takes precedence if specified)
            device: Device (auto, cuda, cpu)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            trust_remote_code: Whether to trust remote code
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
        """Resolve model path from model name or provided path."""
        # If direct path is specified, use it
        if model_path and os.path.exists(model_path):
            return model_path
        
        # Get base path from environment variable (LOCAL_LLM_PATH or fallback)
        base_path = os.getenv("LOCAL_LLM_PATH") or os.getenv("LLM_MODELS_PATH", "/app/llm-models")
        
        # Find in supported models list
        if model_name.lower() in self.SUPPORTED_MODELS:
            model_info = self.SUPPORTED_MODELS[model_name.lower()]
            local_path = os.path.join(base_path, model_info["local_path"])
            
            if os.path.exists(local_path):
                return local_path
            
            # If local path doesn't exist, return HuggingFace model name
            return model_info["name"]
        
        # Treat as direct HuggingFace model name
        return model_name
    
    def load(self):
        """Load the model."""
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
        Generate text.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            max_new_tokens: Maximum number of tokens
            temperature: Sampling temperature
            
        Returns:
            Generated text
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
        Generate text in JSON format.
        
        Args:
            prompt: Prompt (should include JSON output instruction)
            system_prompt: System prompt
            
        Returns:
            Parsed JSON dictionary or None
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
        """Clean up resources."""
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
        """Check if model is loaded."""
        return self._loaded
    
    def __del__(self):
        """Destructor."""
        self.cleanup()


# 전역 인스턴스 (싱글톤 패턴)
_global_llm: Optional[LocalLLM] = None


def get_local_llm(
    model_name: str = "qwen",
    model_path: Optional[str] = None,
    **kwargs
) -> LocalLLM:
    """
    Get global LocalLLM instance (singleton).
    
    Args:
        model_name: Model name
        model_path: Local model path
        **kwargs: Additional LocalLLM arguments
        
    Returns:
        LocalLLM instance
    """
    global _global_llm
    
    if _global_llm is None:
        _global_llm = LocalLLM(model_name=model_name, model_path=model_path, **kwargs)
    
    return _global_llm


def cleanup_global_llm():
    """Clean up global LLM instance."""
    global _global_llm
    
    if _global_llm is not None:
        _global_llm.cleanup()
        _global_llm = None

