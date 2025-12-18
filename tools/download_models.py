#!/usr/bin/env python3
"""
Download Chatterbox TTS ONNX models from HuggingFace.

Downloads the quantized ONNX models required for the C++ TTS demo.

Usage:
    python download_models.py --output-dir models/
    python download_models.py --output-dir models/ --dtype q4
"""

import argparse
import os
import sys

# Model files for each dtype
MODEL_FILES = {
    "speech_encoder": "{dtype}.onnx",
    "embed_tokens": "{dtype}.onnx", 
    "language_model": "{dtype}.onnx",
    "conditional_decoder": "{dtype}.onnx",
}

# Additional files (tokenizer, etc.)
EXTRA_FILES = [
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

REPO_ID = "ResembleAI/chatterbox-turbo-ONNX"


def download_file(repo_id: str, filename: str, local_path: str, token: str = None):
    """Download a single file from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download
    
    # Determine subfolder from filename
    if filename.startswith("onnx/"):
        subfolder = "onnx"
        filename = filename[5:]  # Remove "onnx/" prefix
    else:
        subfolder = None
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            token=token,
        )
        return downloaded_path
    except Exception as e:
        print(f"  Error downloading {filename}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download Chatterbox TTS ONNX models from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_models.py --output-dir models/
  python download_models.py --output-dir models/ --dtype q4
  python download_models.py --output-dir models/ --dtype q8 --token YOUR_HF_TOKEN

Model sizes (approximate):
  fp32: ~3.5 GB total
  q8:   ~1.8 GB total  
  q4:   ~750 MB total (recommended)
"""
    )
    parser.add_argument("-o", "--output-dir", default="models", 
                        help="Output directory for models (default: models/)")
    parser.add_argument("--dtype", default="q4", choices=["fp32", "fp16", "q8", "q4", "q4f16"],
                        help="Model quantization type (default: q4)")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--skip-tokenizer", action="store_true",
                        help="Skip downloading tokenizer files")
    args = parser.parse_args()
    
    # Import huggingface_hub
    try:
        from huggingface_hub import hf_hub_download, HfApi
    except ImportError:
        print("Error: huggingface_hub library not installed.", file=sys.stderr)
        print("Install with: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)
    
    # Get token from args or environment
    token = args.token or os.environ.get("HF_TOKEN")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    onnx_dir = os.path.join(args.output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    
    print(f"Downloading Chatterbox TTS ONNX models ({args.dtype})...")
    print(f"Repository: {REPO_ID}")
    print(f"Output: {args.output_dir}/")
    print()
    
    # Download ONNX model files
    dtype_suffix = f"_{args.dtype}" if args.dtype != "fp32" else ""
    
    model_names = ["speech_encoder", "embed_tokens", "language_model", "conditional_decoder"]
    
    for model_name in model_names:
        # Main .onnx file
        onnx_filename = f"{model_name}{dtype_suffix}.onnx"
        onnx_data_filename = f"{model_name}{dtype_suffix}.onnx_data"
        
        print(f"Downloading {onnx_filename}...")
        result = download_file(REPO_ID, f"onnx/{onnx_filename}", args.output_dir, token)
        if result:
            print(f"  ✓ {onnx_filename}")
        
        # Try to download .onnx_data file (may not exist for all models)
        print(f"Downloading {onnx_data_filename}...")
        result = download_file(REPO_ID, f"onnx/{onnx_data_filename}", args.output_dir, token)
        if result:
            print(f"  ✓ {onnx_data_filename}")
        else:
            print(f"  (no external data file)")
    
    # Download tokenizer files
    if not args.skip_tokenizer:
        print("\nDownloading tokenizer files...")
        for filename in EXTRA_FILES:
            print(f"Downloading {filename}...")
            result = download_file(REPO_ID, filename, args.output_dir, token)
            if result:
                print(f"  ✓ {filename}")
            else:
                print(f"  (optional, skipped)")
    
    print("\n" + "="*50)
    print("Download complete!")
    print(f"Models saved to: {os.path.abspath(args.output_dir)}/")
    print("\nNext steps:")
    print(f"  1. Tokenize text: python tools/pretokenize.py \"Hello!\" -o input.tokens")
    print(f"  2. Run TTS: chatterbox_tts_demo -t input.tokens -m {args.output_dir}")


if __name__ == "__main__":
    main()
