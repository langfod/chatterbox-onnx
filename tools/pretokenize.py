#!/usr/bin/env python3
"""
Pretokenize text for Chatterbox TTS C++ Demo.

This script tokenizes text using the HuggingFace tokenizer and saves
the token IDs in a binary format that the C++ demo can read directly.

Usage:
    python pretokenize.py "Hello world!" -o output.tokens
    python pretokenize.py "Hello world!" -o output.tokens --verbose

Binary format:
    [num_tokens: uint32_le] [token_0: uint32_le] [token_1: uint32_le] ...
"""

import argparse
import struct
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Pretokenize text for Chatterbox TTS C++ demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pretokenize.py "Hello world!" -o hello.tokens
  python pretokenize.py "This is a longer sentence." -o input.tokens --verbose
  
Then run C++ demo:
  chatterbox_tts_demo -t hello.tokens -o output.wav
"""
    )
    parser.add_argument("text", help="Text to tokenize")
    parser.add_argument("-o", "--output", required=True, help="Output token file path (.tokens)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed token information")
    parser.add_argument("--model", default="ResembleAI/chatterbox-turbo-ONNX", 
                        help="HuggingFace model ID for tokenizer (default: ResembleAI/chatterbox-turbo-ONNX)")
    args = parser.parse_args()
    
    # Import transformers here to give better error message if not installed
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: transformers library not installed.", file=sys.stderr)
        print("Install with: pip install transformers", file=sys.stderr)
        sys.exit(1)
    
    # Load tokenizer
    if args.verbose:
        print(f"Loading tokenizer from {args.model}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Tokenize text (no special tokens - those are handled by the model)
    tokens = tokenizer.encode(args.text, add_special_tokens=False)
    
    if len(tokens) == 0:
        print("Warning: Text produced no tokens!", file=sys.stderr)
    
    # Write binary format: [num_tokens (uint32)] [token_ids (uint32 each)]
    try:
        with open(args.output, "wb") as f:
            # Write number of tokens (little-endian uint32)
            f.write(struct.pack("<I", len(tokens)))
            # Write each token ID (little-endian uint32)
            for tok in tokens:
                f.write(struct.pack("<I", tok))
    except IOError as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print summary
    print(f"Wrote {len(tokens)} tokens to {args.output}")
    
    if args.verbose:
        print(f"\nInput text: \"{args.text}\"")
        print(f"Token IDs:  {tokens}")
        
        # Decode back to verify
        decoded = tokenizer.decode(tokens)
        print(f"Decoded:    \"{decoded}\"")
        
        # Show individual tokens
        print("\nToken breakdown:")
        for i, tok in enumerate(tokens):
            tok_str = tokenizer.decode([tok])
            print(f"  [{i:3d}] {tok:6d} -> \"{tok_str}\"")


if __name__ == "__main__":
    main()
