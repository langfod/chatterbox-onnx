#!/usr/bin/env python3
"""
Analyze ONNX Runtime profiling JSON files.

Usage:
    python analyze_onnx_profile.py onnx_profile_*.json
    python analyze_onnx_profile.py onnx_profile_*.json --top 20
    python analyze_onnx_profile.py onnx_profile_*.json --by-type
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict


def load_profile(filepath: str) -> list:
    """Load ONNX Runtime profiling JSON file.
    
    Handles the case where multiple sessions write concatenated JSON arrays,
    including partially corrupted boundaries between arrays.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Try standard JSON first
    content = ''.join(lines)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # ONNX Runtime writes multiple JSON arrays concatenated together
    # Find array boundaries by looking for lines that are just "]"
    all_events = []
    array_start = 0
    array_count = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == ']':
            # Found end of an array, try to parse from array_start to here
            array_content = ''.join(lines[array_start:i+1])
            try:
                events = json.loads(array_content)
                if isinstance(events, list):
                    all_events.extend(events)
                    array_count += 1
                    print(f"  Parsed array {array_count}: {len(events)} events (lines {array_start+1}-{i+1})")
            except json.JSONDecodeError as e:
                # Try to recover by finding the first valid JSON object line
                print(f"  Array at lines {array_start+1}-{i+1} failed: {e}")
                print(f"  Attempting recovery...")
                
                # Find first line that looks like a JSON object
                for j in range(array_start, i+1):
                    if lines[j].strip().startswith('{"cat"'):
                        # Build array from this point
                        recovered_lines = ['[\n'] + lines[j:i+1]
                        recovered_content = ''.join(recovered_lines)
                        try:
                            events = json.loads(recovered_content)
                            if isinstance(events, list):
                                all_events.extend(events)
                                array_count += 1
                                print(f"  Recovered array {array_count}: {len(events)} events (lines {j+1}-{i+1})")
                                break
                        except json.JSONDecodeError:
                            continue
            array_start = i + 1
    
    if not all_events:
        raise ValueError(f"Could not parse any JSON from {filepath}")
    
    print(f"Total: {len(all_events)} events from {array_count} session profiles")
    return all_events


def analyze_profile(events: list, top_n: int = 15, group_by_type: bool = False):
    """Analyze profiling events and print summary."""
    
    # Filter to kernel/operator events (have 'dur' field for duration)
    ops = []
    for event in events:
        # Skip non-dict items (can happen with nested structures)
        if not isinstance(event, dict):
            continue
        if 'dur' in event and 'name' in event:
            ops.append({
                'name': event['name'],
                'dur_us': event['dur'],  # duration in microseconds
                'cat': event.get('cat', 'unknown'),
                'args': event.get('args', {})
            })
    
    if not ops:
        print("No operator timing events found in profile.")
        return
    
    # Calculate total time
    total_us = sum(op['dur_us'] for op in ops)
    total_ms = total_us / 1000.0
    
    print(f"\n{'='*70}")
    print(f"ONNX Runtime Profile Summary")
    print(f"{'='*70}")
    print(f"Total events: {len(ops)}")
    print(f"Total time: {total_ms:.2f} ms ({total_us} µs)")
    print()
    
    if group_by_type:
        # Group by operator type
        by_type = defaultdict(lambda: {'count': 0, 'total_us': 0, 'ops': []})
        
        for op in ops:
            # Extract op type from name (e.g., "/layer/MatMul" -> "MatMul")
            name = op['name']
            op_type = op['args'].get('op_name', name.split('/')[-1].split('_')[0])
            
            by_type[op_type]['count'] += 1
            by_type[op_type]['total_us'] += op['dur_us']
            by_type[op_type]['ops'].append(op)
        
        # Sort by total time
        sorted_types = sorted(by_type.items(), key=lambda x: x[1]['total_us'], reverse=True)
        
        print(f"{'Operator Type':<30} {'Count':>8} {'Total (ms)':>12} {'Avg (ms)':>10} {'%':>8}")
        print('-' * 70)
        
        for op_type, data in sorted_types[:top_n]:
            count = data['count']
            total = data['total_us'] / 1000.0
            avg = total / count if count > 0 else 0
            pct = (data['total_us'] / total_us * 100) if total_us > 0 else 0
            print(f"{op_type:<30} {count:>8} {total:>12.3f} {avg:>10.3f} {pct:>7.1f}%")
        
    else:
        # Sort by duration (slowest first)
        sorted_ops = sorted(ops, key=lambda x: x['dur_us'], reverse=True)
        
        print(f"Top {top_n} Slowest Operations:")
        print(f"{'Operation':<50} {'Time (ms)':>12} {'%':>8}")
        print('-' * 70)
        
        for op in sorted_ops[:top_n]:
            dur_ms = op['dur_us'] / 1000.0
            pct = (op['dur_us'] / total_us * 100) if total_us > 0 else 0
            name = op['name'][:48] + '..' if len(op['name']) > 50 else op['name']
            print(f"{name:<50} {dur_ms:>12.3f} {pct:>7.1f}%")
    
    # Show session-level events
    print()
    print("Session Events:")
    print('-' * 70)
    session_events = [op for op in ops if op['cat'] == 'Session']
    for op in session_events:
        dur_ms = op['dur_us'] / 1000.0
        print(f"  {op['name']:<45} {dur_ms:>10.3f} ms")
    
    # Summary by category
    print()
    print("By Category:")
    print('-' * 70)
    by_cat = defaultdict(lambda: {'count': 0, 'total_us': 0})
    for op in ops:
        by_cat[op['cat']]['count'] += 1
        by_cat[op['cat']]['total_us'] += op['dur_us']
    
    for cat, data in sorted(by_cat.items(), key=lambda x: x[1]['total_us'], reverse=True):
        total = data['total_us'] / 1000.0
        pct = (data['total_us'] / total_us * 100) if total_us > 0 else 0
        print(f"  {cat:<45} {total:>10.3f} ms ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Analyze ONNX Runtime profiling JSON files')
    parser.add_argument('files', nargs='+', help='Profile JSON file(s)')
    parser.add_argument('--top', type=int, default=15, help='Number of top operations to show (default: 15)')
    parser.add_argument('--by-type', action='store_true', help='Group results by operator type')
    args = parser.parse_args()
    
    for filepath in args.files:
        path = Path(filepath)
        if not path.exists():
            print(f"File not found: {filepath}")
            continue
        
        print(f"\nAnalyzing: {filepath}")
        events = load_profile(filepath)
        analyze_profile(events, top_n=args.top, group_by_type=args.by_type)


if __name__ == '__main__':
    main()
