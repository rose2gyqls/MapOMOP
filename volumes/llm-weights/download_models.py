#!/usr/bin/env python3
"""
HuggingFace Model Download Script

Downloads LLM models from HuggingFace and saves them locally.
Supports downloading multiple models or individual models via command line.

Usage:
    # Download all models
    python download_models.py

    # Download specific model
    python download_models.py --model gemma-2-2b

    # Download to custom directory
    python download_models.py --output-dir /path/to/models
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: transformers library not installed.")
    print("Install with: pip install transformers torch")
    sys.exit(1)


# Model configurations
MODEL_CONFIGS: List[Dict[str, any]] = [
    {
        "id": "gemma-2-2b",
        "name": "google/gemma-2-2b",
        "trust_remote_code": False,
        "description": "Google Gemma 2 2B model"
    },
    {
        "id": "hari-q3-8b",
        "name": "snuh/hari-q3-8b",
        "trust_remote_code": True,
        "description": "SNUH Hari Q3 8B model"
    },
    {
        "id": "qwen2.5-3b",
        "name": "Qwen/Qwen2.5-3B",
        "trust_remote_code": True,
        "description": "Qwen2.5 3B model"
    }
]


def get_model_config(model_id: str) -> Optional[Dict[str, any]]:
    """Get model configuration by ID."""
    for config in MODEL_CONFIGS:
        if config["id"] == model_id:
            return config
    return None


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
    trust_remote_code: bool = False,
    device: str = "auto"
) -> bool:
    """
    Download model from HuggingFace and save to local directory.

    Args:
        model_name: HuggingFace model identifier
        output_dir: Local directory to save the model
        trust_remote_code: Whether to trust remote code in model config
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
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        tokenizer.save_pretrained(output_path)
        print("✓ Tokenizer saved")

        # Download model
        print("Downloading model... (this may take a while)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            dtype="auto",
            low_cpu_mem_usage=True,
            device_map=device
        )
        model.save_pretrained(output_path)
        print("✓ Model saved")

        # Calculate and display size
        total_size = calculate_directory_size(output_path)
        print(f"Total size: {format_size(total_size)}")

        return True

    except Exception as e:
        print(f"✗ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_available_models():
    """List all available model configurations."""
    print("Available models:")
    print("-" * 60)
    for i, config in enumerate(MODEL_CONFIGS, 1):
        print(f"  {i}. {config['id']}")
        print(f"     Name: {config['name']}")
        print(f"     Description: {config['description']}")
        print()


def list_downloaded_models(base_dir: Path):
    """List all downloaded models in the base directory."""
    print("\n" + "="*60)
    print("Downloaded Models")
    print("="*60)
    
    models_found = False
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and d.name != "__pycache__":
            size = calculate_directory_size(d)
            print(f"  {d.name}: {format_size(size)}")
            models_found = True
    
    if not models_found:
        print("  No models found.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download HuggingFace models locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models
  python download_models.py

  # Download specific model
  python download_models.py --model gemma-2-2b

  # Download to custom directory
  python download_models.py --output-dir /path/to/models

  # List available models
  python download_models.py --list
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Model ID to download (e.g., gemma-2-2b). If not specified, downloads all models."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory to save models (default: script directory)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to load model on (default: auto)"
    )

    args = parser.parse_args()

    # List models and exit
    if args.list:
        list_available_models()
        return

    # Determine base directory
    base_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent

    # Get models to download
    if args.model:
        # Download specific model
        config = get_model_config(args.model)
        if not config:
            print(f"Error: Model '{args.model}' not found.")
            print("\nAvailable models:")
            for cfg in MODEL_CONFIGS:
                print(f"  - {cfg['id']}")
            sys.exit(1)
        
        models_to_download = [config]
    else:
        # Download all models
        models_to_download = MODEL_CONFIGS

    # Display download plan
    print("="*60)
    print("HuggingFace Model Download")
    print("="*60)
    print(f"\nBase directory: {base_dir.absolute()}")
    print(f"\nModels to download ({len(models_to_download)}):")
    for i, model_info in enumerate(models_to_download, 1):
        print(f"  {i}. {model_info['name']} ({model_info['id']})")
        print(f"     → {model_info['description']}")

    # Download models
    results = {}
    for model_info in models_to_download:
        output_dir = base_dir / model_info["id"]
        success = download_model(
            model_name=model_info["name"],
            output_dir=str(output_dir),
            trust_remote_code=model_info["trust_remote_code"],
            device=args.device
        )
        results[model_info["name"]] = success

    # Display results
    print("\n" + "="*60)
    print("Download Results")
    print("="*60)
    all_success = True
    for name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {name}: {status}")
        if not success:
            all_success = False

    # List downloaded models
    list_downloaded_models(base_dir)

    # Exit with appropriate code
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
