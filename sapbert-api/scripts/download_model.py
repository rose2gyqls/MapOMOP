#!/usr/bin/env python3
"""
SapBERT 모델 사전 다운로드 스크립트

오프라인 환경(병원 내부망)에서 사용하기 위해
모델을 미리 다운로드하여 저장합니다.

사용법:
    python download_model.py [--output-dir ./sapbert-model]
"""

import argparse
import os
import sys
from pathlib import Path


def download_model(model_name: str, output_dir: str) -> bool:
    """
    HuggingFace에서 모델을 다운로드하여 로컬에 저장합니다.
    
    Args:
        model_name: HuggingFace 모델명
        output_dir: 저장 디렉토리
        
    Returns:
        성공 여부
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("Error: transformers 라이브러리가 설치되지 않았습니다.")
        print("설치: pip install transformers torch")
        return False
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"모델 다운로드 중: {model_name}")
    print(f"저장 경로: {output_path.absolute()}")
    print("-" * 50)
    
    try:
        # Tokenizer 다운로드
        print("Tokenizer 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_path)
        print("✓ Tokenizer 저장 완료")
        
        # Model 다운로드
        print("Model 다운로드 중...")
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(output_path)
        print("✓ Model 저장 완료")
        
        # 파일 목록 출력
        print("-" * 50)
        print("저장된 파일:")
        total_size = 0
        for file in output_path.iterdir():
            size = file.stat().st_size
            total_size += size
            size_mb = size / (1024 * 1024)
            print(f"  {file.name}: {size_mb:.2f} MB")
        
        print("-" * 50)
        print(f"총 크기: {total_size / (1024 * 1024):.2f} MB")
        print(f"\n모델이 {output_path.absolute()}에 저장되었습니다.")
        print("\n오프라인 환경에서 사용하려면:")
        print(f"  SAPBERT_MODEL_PATH={output_path.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"Error: 모델 다운로드 실패 - {e}")
        return False


def verify_model(model_path: str) -> bool:
    """
    저장된 모델이 정상적으로 로드되는지 확인합니다.
    
    Args:
        model_path: 모델 경로
        
    Returns:
        검증 성공 여부
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("Error: transformers 라이브러리가 필요합니다.")
        return False
    
    print(f"\n모델 검증 중: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        
        # 테스트 임베딩 생성
        import torch
        
        test_text = "myocardial ischemia"
        inputs = tokenizer(
            test_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=25
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
        
        print(f"✓ 모델 로드 성공")
        print(f"✓ 테스트 텍스트: '{test_text}'")
        print(f"✓ 임베딩 차원: {embedding.shape[1]}")
        print(f"✓ 검증 완료")
        
        return True
        
    except Exception as e:
        print(f"✗ 모델 검증 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='SapBERT 모델 다운로드 스크립트'
    )
    parser.add_argument(
        '--model-name',
        default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
        help='HuggingFace 모델명'
    )
    parser.add_argument(
        '--output-dir',
        default='./sapbert-model',
        help='모델 저장 디렉토리'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='다운로드 후 모델 검증'
    )
    parser.add_argument(
        '--verify-only',
        type=str,
        metavar='PATH',
        help='기존 모델 검증만 수행'
    )
    
    args = parser.parse_args()
    
    # 검증만 수행
    if args.verify_only:
        success = verify_model(args.verify_only)
        sys.exit(0 if success else 1)
    
    # 모델 다운로드
    success = download_model(args.model_name, args.output_dir)
    
    if not success:
        sys.exit(1)
    
    # 검증
    if args.verify:
        success = verify_model(args.output_dir)
        sys.exit(0 if success else 1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()

