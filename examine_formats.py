#!/usr/bin/env python3
import json

def examine_file(filepath, num_lines=3):
    print(f"\n=== Examining {filepath} ===")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i in range(num_lines):
                line = f.readline()
                if not line:
                    break
                try:
                    data = json.loads(line.strip())
                    print(f"Line {i+1} structure:")
                    print(f"  Keys: {list(data.keys())}")
                    for key, value in data.items():
                        if isinstance(value, str):
                            print(f"  {key}: '{value[:100]}...' (length: {len(value)})")
                        elif isinstance(value, list):
                            print(f"  {key}: list with {len(value)} items")
                            if value and isinstance(value[0], dict):
                                print(f"    First item keys: {list(value[0].keys())}")
                        else:
                            print(f"  {key}: {type(value).__name__}")
                    print()
                except json.JSONDecodeError as e:
                    print(f"  JSON decode error on line {i+1}: {e}")
                    print(f"  Raw line: {line[:200]}...")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

if __name__ == "__main__":
    examine_file("data/lila_filtered/lila_filtered.jsonl")
    examine_file("data/gsm8k/gsm8k.jsonl")