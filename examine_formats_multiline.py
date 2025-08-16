#!/usr/bin/env python3
import json

def examine_json_file(filepath):
    print(f"\n=== Examining {filepath} ===")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Top-level keys: {list(data.keys())}")
        
        for key, value in data.items():
            if isinstance(value, str):
                print(f"  {key}: '{value[:100]}...' (length: {len(value)})")
            elif isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
                if value:
                    first_item = value[0]
                    if isinstance(first_item, dict):
                        print(f"    First item keys: {list(first_item.keys())}")
                        for sub_key, sub_value in first_item.items():
                            if isinstance(sub_value, str):
                                print(f"      {sub_key}: '{sub_value[:100]}...' (length: {len(sub_value)})")
                            else:
                                print(f"      {sub_key}: {type(sub_value).__name__}")
                    else:
                        print(f"    First item: {type(first_item).__name__}")
            else:
                print(f"  {key}: {type(value).__name__}")
                
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

if __name__ == "__main__":
    examine_json_file("data/lila_filtered/lila_filtered.jsonl")
    examine_json_file("data/gsm8k/gsm8k.jsonl")