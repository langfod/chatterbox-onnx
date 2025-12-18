#!/usr/bin/env python3
"""
Batch tokenize multiple lines from a text file.

This script reads a text file with one utterance per line and creates
individual .tokens files for each line.

Usage:
    python batch_tokenize.py input.txt -o tokens_dir/
"""

import argparse
import struct
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="Batch pretokenize text file for Chatterbox TTS C++ demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_tokenize.py dialogue.txt -o tokens/
  python batch_tokenize.py sentences.txt -o output_tokens/ --prefix scene1
  
Input file format (one sentence per line):
  Hello, how are you?
  I'm doing great, thanks!
  See you later.

Output files:
  tokens/line_0000.tokens
  tokens/line_0001.tokens
  tokens/line_0002.tokens
"""
    )
    parser.add_argument("input", help="Input text file (one line per utterance)")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory for .tokens files")
    parser.add_argument("--prefix", default="line", help="Prefix for output files (default: line)")
    parser.add_argument("--model", default="ResembleAI/chatterbox",
                        help="HuggingFace model ID for tokenizer (default: ResembleAI/chatterbox)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print each line being processed")
    args = parser.parse_args()
    
    # Import transformers
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Error: transformers library not installed.", file=sys.stderr)
        print("Install with: pip install transformers", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Read input file
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except IOError as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not lines:
        print("Warning: Input file is empty or contains only blank lines.")
        sys.exit(0)
    
    print(f"Processing {len(lines)} lines...")
    
    # Process each line
    total_tokens = 0
    for i, line in enumerate(lines):
        tokens = tokenizer.encode(line, add_special_tokens=False)
        total_tokens += len(tokens)
        
        output_path = os.path.join(args.output_dir, f"{args.prefix}_{i:04d}.tokens")
        
        with open(output_path, "wb") as f:
            f.write(struct.pack("<I", len(tokens)))
            for tok in tokens:
                f.write(struct.pack("<I", tok))
        
        if args.verbose:
            preview = line[:50] + "..." if len(line) > 50 else line
            print(f"  [{i+1:4d}/{len(lines)}] {len(tokens):4d} tokens: \"{preview}\"")
        else:
            # Progress indicator
            if (i + 1) % 100 == 0 or i == len(lines) - 1:
                print(f"  Processed {i+1}/{len(lines)} lines...")
    
    print(f"\nDone! Created {len(lines)} token files in {args.output_dir}/")
    print(f"Total tokens: {total_tokens} (avg: {total_tokens/len(lines):.1f} per line)")


if __name__ == "__main__":
    main()
