#!/usr/bin/env python3
"""
sLLM 모델 다운로드 스크립트
로컬 구동 가능한 작은 모델들을 다운로드합니다.

모델 목록:
1. Qwen2.5-1.5B-Instruct (약 3GB)
2. Llama-3.2-1B-Instruct (약 2.5GB) 
3. SNUH/Hari 모델 (HuggingFace에서 확인 필요)
"""

import os
import sys
from pathlib import Path

def download_model(model_name: str, output_dir: str, trust_remote_code: bool = False):
    """HuggingFace에서 모델 다운로드"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"다운로드 중: {model_name}")
    print(f"저장 경로: {output_path}")
    print('='*60)
    
    try:
        # Tokenizer 다운로드
        print("Tokenizer 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        tokenizer.save_pretrained(output_path)
        print("✓ Tokenizer 저장 완료")
        
        # Model 다운로드
        print("Model 다운로드 중... (시간이 걸릴 수 있습니다)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )
        model.save_pretrained(output_path)
        print("✓ Model 저장 완료")
        
        # 파일 크기 확인
        total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
        print(f"총 크기: {total_size / (1024**3):.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"✗ 다운로드 실패: {e}")
        return False


def main():
    base_dir = Path(__file__).parent
    
    # 다운로드할 모델 목록
    models = [
        {
            "name": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_dir": base_dir / "qwen2.5-0.5b-instruct",
            "trust_remote_code": True,
            "description": "Qwen2.5 0.5B - 가장 작은 Qwen 모델"
        },
        {
            "name": "meta-llama/Llama-3.2-1B-Instruct",
            "output_dir": base_dir / "llama-3.2-1b-instruct",
            "trust_remote_code": False,
            "description": "Llama 3.2 1B - 작은 Llama 모델"
        },
        {
            "name": "beomi/Llama-3-Open-Ko-8B-Instruct-preview",
            "output_dir": base_dir / "llama-3-ko-8b-instruct",
            "trust_remote_code": False,
            "description": "Korean Llama 3 8B (대체 모델, snuh/hari 없을 경우)"
        }
    ]
    
    print("="*60)
    print("sLLM 모델 다운로드")
    print("="*60)
    print("\n다운로드할 모델:")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m['name']}")
        print(f"     → {m['description']}")
    
    results = {}
    
    for model_info in models:
        success = download_model(
            model_name=model_info["name"],
            output_dir=str(model_info["output_dir"]),
            trust_remote_code=model_info["trust_remote_code"]
        )
        results[model_info["name"]] = success
    
    # 결과 출력
    print("\n" + "="*60)
    print("다운로드 결과")
    print("="*60)
    for name, success in results.items():
        status = "✓ 성공" if success else "✗ 실패"
        print(f"  {name}: {status}")
    
    # 디렉토리 내용 출력
    print("\n" + "="*60)
    print("다운로드된 모델 디렉토리")
    print("="*60)
    for d in base_dir.iterdir():
        if d.is_dir() and d.name != "__pycache__":
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            print(f"  {d.name}: {size / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()

