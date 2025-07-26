import json
import os

def fix_gsm8k_format():
    """
    More robust script to convert data/gsm8k.jsonl in-place.
    Handles potential file issues and provides better error reporting.
    """
    input_file = "data/gsm8k.jsonl"
    temp_file = "data/gsm8k_temp.jsonl"
    
    converted_count = 0
    line_count = 0
    
    print(f"Starting conversion for {input_file}...")
    
    try:
        # Check if file exists and is not empty
        if not os.path.exists(input_file) or os.path.getsize(input_file) == 0:
            print(f"Error: The file {input_file} is missing or empty.")
            return

        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(temp_file, 'w', encoding='utf-8') as fout:
            
            for line_num, line in enumerate(fin, 1):
                line_count += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    
                    if isinstance(data, dict) and 'question' in data and 'answer' in data:
                        combined_text = f"Question: {data['question']}\nAnswer: {data['answer']}"
                        new_data = {"text": combined_text}
                        json.dump(new_data, fout, ensure_ascii=False)
                        fout.write('\n')
                        converted_count += 1
                    else:
                        print(f"Line {line_num}: Does not contain 'question' and 'answer'. Keeping original.")
                        json.dump(data, fout, ensure_ascii=False)
                        fout.write('\n')
                        
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error on line {line_num}: {e}")
                    print(f"--> Offending line: '{line}'")
                    continue
        
        os.replace(temp_file, input_file)
        
        print("\n--- Conversion Summary ---")
        print(f"Total lines processed: {line_count}")
        print(f"Lines successfully converted: {converted_count}")
        print(f"File '{input_file}' has been updated.")
        
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print("Temporary file cleaned up.")

if __name__ == "__main__":
    fix_gsm8k_format() 