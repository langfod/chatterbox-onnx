#!/usr/bin/env python3
"""
convert_cond.py - Read and convert Chatterbox .cond cache files

This tool reads the binary .cond format used by the C++ Chatterbox TTS
for caching voice conditionals, and can convert them to other formats.

Usage:
    python convert_cond.py <input.cond>                    # Display info
    python convert_cond.py <input.cond> -o output.npz      # Convert to npz
    python convert_cond.py <input.cond> --torch output.pt  # Convert to PyTorch
"""

import argparse
import struct
import sys
from pathlib import Path
import numpy as np

# .cond file format:
# Header:
#   uint32 magic   = 0x434F4E44 ("COND")
#   uint32 version = 1
#
# For each tensor (condEmb, promptToken, speakerEmbeddings, speakerFeatures):
#   uint32 numDims
#   int64[numDims] shape
#   uint64 dataSize (bytes)
#   data[dataSize]

COND_MAGIC = 0x434F4E44  # "COND"
COND_VERSION = 1


def read_cond_file(filepath: str) -> dict:
    """Read a .cond file and return tensors as numpy arrays."""
    
    with open(filepath, 'rb') as f:
        # Read header
        magic, version = struct.unpack('<II', f.read(8))
        
        if magic != COND_MAGIC:
            raise ValueError(f"Invalid magic number: 0x{magic:08X}, expected 0x{COND_MAGIC:08X}")
        
        if version != COND_VERSION:
            raise ValueError(f"Unsupported version: {version}, expected {COND_VERSION}")
        
        def read_float_array():
            num_dims = struct.unpack('<I', f.read(4))[0]
            shape = struct.unpack(f'<{num_dims}q', f.read(8 * num_dims))
            data_size = struct.unpack('<Q', f.read(8))[0]
            data = np.frombuffer(f.read(data_size), dtype=np.float32)
            return data.reshape(shape)
        
        def read_int64_array():
            num_dims = struct.unpack('<I', f.read(4))[0]
            shape = struct.unpack(f'<{num_dims}q', f.read(8 * num_dims))
            data_size = struct.unpack('<Q', f.read(8))[0]
            data = np.frombuffer(f.read(data_size), dtype=np.int64)
            return data.reshape(shape)
        
        result = {
            'cond_emb': read_float_array(),
            'prompt_token': read_int64_array(),
            'speaker_embeddings': read_float_array(),
            'speaker_features': read_float_array(),
        }
        
        return result


def write_cond_file(filepath: str, data: dict):
    """Write tensors to a .cond file."""
    
    with open(filepath, 'wb') as f:
        # Write header
        f.write(struct.pack('<II', COND_MAGIC, COND_VERSION))
        
        def write_float_array(arr):
            arr = np.asarray(arr, dtype=np.float32)
            shape = arr.shape
            f.write(struct.pack('<I', len(shape)))
            f.write(struct.pack(f'<{len(shape)}q', *shape))
            data_bytes = arr.tobytes()
            f.write(struct.pack('<Q', len(data_bytes)))
            f.write(data_bytes)
        
        def write_int64_array(arr):
            arr = np.asarray(arr, dtype=np.int64)
            shape = arr.shape
            f.write(struct.pack('<I', len(shape)))
            f.write(struct.pack(f'<{len(shape)}q', *shape))
            data_bytes = arr.tobytes()
            f.write(struct.pack('<Q', len(data_bytes)))
            f.write(data_bytes)
        
        write_float_array(data['cond_emb'])
        write_int64_array(data['prompt_token'])
        write_float_array(data['speaker_embeddings'])
        write_float_array(data['speaker_features'])


def print_info(data: dict, filepath: str):
    """Print information about the loaded conditionals."""
    
    print(f"\n=== Chatterbox Voice Conditionals: {filepath} ===\n")
    
    for name, arr in data.items():
        size_kb = arr.nbytes / 1024
        print(f"  {name}:")
        print(f"    Shape: {arr.shape}")
        print(f"    Dtype: {arr.dtype}")
        print(f"    Size:  {size_kb:.2f} KB")
        if arr.dtype in [np.float32, np.float64]:
            print(f"    Range: [{arr.min():.4f}, {arr.max():.4f}]")
        print()
    
    total_kb = sum(arr.nbytes for arr in data.values()) / 1024
    print(f"  Total size: {total_kb:.2f} KB\n")


def convert_to_npz(data: dict, output_path: str):
    """Save to NumPy .npz format."""
    np.savez(output_path, **data)
    print(f"Saved to: {output_path}")


def convert_to_torch(data: dict, output_path: str):
    """Save to PyTorch .pt format."""
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not installed. Install with: pip install torch")
        sys.exit(1)
    
    torch_data = {k: torch.from_numpy(v) for k, v in data.items()}
    torch.save(torch_data, output_path)
    print(f"Saved to: {output_path}")


def convert_from_npz(npz_path: str, output_path: str):
    """Convert from .npz to .cond format."""
    data = dict(np.load(npz_path))
    write_cond_file(output_path, data)
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Read and convert Chatterbox .cond cache files"
    )
    parser.add_argument('input', help='Input .cond or .npz file')
    parser.add_argument('-o', '--output', help='Output file path (.npz)')
    parser.add_argument('--torch', metavar='PATH', help='Output as PyTorch .pt file')
    parser.add_argument('--to-cond', metavar='PATH', 
                       help='Convert .npz to .cond format')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    # Handle .npz to .cond conversion
    if args.to_cond:
        if input_path.suffix == '.npz':
            convert_from_npz(args.input, args.to_cond)
        else:
            print("Error: --to-cond requires a .npz input file")
            sys.exit(1)
        return
    
    # Read .cond file
    if input_path.suffix != '.cond':
        print(f"Warning: Expected .cond file, got {input_path.suffix}")
    
    try:
        data = read_cond_file(args.input)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Convert or display info
    if args.output:
        convert_to_npz(data, args.output)
    elif args.torch:
        convert_to_torch(data, args.torch)
    else:
        print_info(data, args.input)


if __name__ == '__main__':
    main()
