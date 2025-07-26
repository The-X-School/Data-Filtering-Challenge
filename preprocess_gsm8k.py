import json
import sys
import os

def convert_gsm8k_to_text_format(input_file, output_file):
    """
    Convert GSM8K format with 'question' and 'answer' fields 
    to format with 'text' field that format_data.py can process.
    """
    instances = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as fin:
            for line_num, line in enumerate(fin, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        
                        # Check if it has question and answer fields
                        if isinstance(data, dict) and 'question' in data and 'answer' in data:
                            # Combine question and answer into a single text field
                            # Format: "Question: {question}\nAnswer: {answer}"
                            combined_text = f"Question: {data['question']}\nAnswer: {data['answer']}"
                            instances.append({"text": combined_text})
                        else:
                            print(f"Warning: Line {line_num} doesn't have expected 'question' and 'answer' fields: {data}")
                            
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
        
        # Write instances in JSONL format (one JSON object per line)
        with open(output_file, 'w', encoding='utf-8') as fout:
            for instance in instances:
                json.dump(instance, fout, ensure_ascii=False)
                fout.write('\n')
        
        print(f"Successfully converted {len(instances)} instances from {input_file} to {output_file}")
        print(f"Format: GSM8K (question/answer) -> Text format for format_data.py")
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {os.path.basename(__file__)} <input_gsm8k_file.jsonl> <output_text_format_file.jsonl>")
        print("Example: python preprocess_gsm8k.py raw_gsm8k.jsonl gsm8k_text_format.jsonl")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist")
        sys.exit(1)
    
    convert_gsm8k_to_text_format(input_path, output_path) 