#!/usr/bin/env python3
"""
SapBERT Model Download Script

Downloads SapBERT model from HuggingFace and saves it locally.
Designed for offline environments (e.g., hospital internal networks).

Usage:
    # Download model to default location (volumes/sapbert_models)
    python download_model.py

    # Download to custom directory
    python download_model.py --output-dir ./custom-path

    # Download and verify
    python download_model.py --verify

    # Verify existing model only
    python download_model.py --verify-only /path/to/model
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from transformers import AutoModel, AutoTokenizer
    import torch
except ImportError:
    print("Error: Required libraries not installed.")
    print("Install with: pip install transformers torch")
    sys.exit(1)


def get_default_output_dir() -> Path:
    """
    Get default output directory relative to project root.
    
    Returns:
        Path to volumes/sapbert_models directory
    """
    # Script is in sapbert-api/scripts/, so go up to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    return project_root / "volumes" / "sapbert_models"


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def calculate_directory_size(directory: Path) -> int:
    """Calculate total size of all files in directory."""
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())


def download_model(
    model_name: str,
    output_dir: str,
    device: str = "auto"
) -> bool:
    """
    Download model from HuggingFace and save to local directory.

    Args:
        model_name: HuggingFace model identifier
        output_dir: Local directory to save the model
        device: Device to load model on (auto, cpu, cuda)

    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Output path: {output_path.absolute()}")
    print('='*60)

    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_path)
        print("✓ Tokenizer saved")

        # Download model
        print("Downloading model... (this may take a while)")
        model = AutoModel.from_pretrained(
            model_name,
            device_map=device if device != "auto" else None
        )
        model.save_pretrained(output_path)
        print("✓ Model saved")

        # List downloaded files
        print("\n" + "-" * 60)
        print("Downloaded files:")
        total_size = 0
        for file in sorted(output_path.iterdir()):
            if file.is_file():
                size = file.stat().st_size
                total_size += size
                print(f"  {file.name}: {format_size(size)}")

        print("-" * 60)
        print(f"Total size: {format_size(total_size)}")
        print(f"\nModel saved to: {output_path.absolute()}")
        print("\nFor offline usage, set environment variable:")
        print(f"  export SAPBERT_MODEL_PATH={output_path.absolute()}")

        return True

    except Exception as e:
        print(f"✗ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_model(model_path: str) -> bool:
    """
    Verify that a saved model can be loaded and used correctly.

    Args:
        model_path: Path to the model directory

    Returns:
        True if verification successful, False otherwise
    """
    model_path_obj = Path(model_path)
    
    if not model_path_obj.exists():
        print(f"✗ Model path does not exist: {model_path}")
        return False

    print(f"\n{'='*60}")
    print(f"Verifying model: {model_path}")
    print('='*60)

    try:
        # Load tokenizer and model
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✓ Tokenizer loaded")

        print("Loading model...")
        model = AutoModel.from_pretrained(model_path)
        print("✓ Model loaded")

        # Test embedding generation
        print("\nTesting embedding generation...")
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

        print(f"✓ Test text: '{test_text}'")
        print(f"✓ Embedding dimension: {embedding.shape[1]}")
        print(f"✓ Embedding shape: {embedding.shape}")
        print("\n✓ Verification successful")

        return True

    except Exception as e:
        print(f"✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download SapBERT model from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to default location (volumes/sapbert_models)
  python download_model.py

  # Download to custom directory
  python download_model.py --output-dir ./my-models

  # Download and verify
  python download_model.py --verify

  # Verify existing model only
  python download_model.py --verify-only /path/to/model
        """
    )

    parser.add_argument(
        '--model-name',
        type=str,
        default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
        help='HuggingFace model identifier (default: cambridgeltl/SapBERT-from-PubMedBERT-fulltext)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for model (default: volumes/sapbert_models)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify model after download'
    )
    parser.add_argument(
        '--verify-only',
        type=str,
        metavar='PATH',
        help='Verify existing model only (skip download)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to load model on (default: auto)'
    )

    args = parser.parse_args()

    # Verify only mode
    if args.verify_only:
        success = verify_model(args.verify_only)
        sys.exit(0 if success else 1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = get_default_output_dir()

    # Download model
    success = download_model(
        model_name=args.model_name,
        output_dir=str(output_dir),
        device=args.device
    )

    if not success:
        sys.exit(1)

    # Verify if requested
    if args.verify:
        success = verify_model(str(output_dir))
        sys.exit(0 if success else 1)

    sys.exit(0)


if __name__ == "__main__":
    main()
