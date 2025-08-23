import os
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

def detokenize_jsonl(input_path, output_path):
    """
    Reads a JSONL file, decodes the 'tokens' field (which is expected to be a list of token IDs),
    and writes the modified data with a new 'text' field to a new JSONL file.

    Args:
        input_path (str): The path to the input JSONL file.
        output_path (str): The path where the output JSONL file will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)

    try:
        # First, count the number of lines to set up the progress bar
        with open(input_path, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for line in f)

        # Now, process the file
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            progress_bar = tqdm(infile, total=num_lines, desc="Detokenizing")
            
            for line in progress_bar:
                try:
                    data = json.loads(line)
                    
                    # Check if 'tokens' is a list and decode it
                    if 'tokens' in data and isinstance(data['tokens'], list):
                        text = tokenizer.decode(
                            data['tokens'], 
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )
                        
                        # Create a new dictionary with only the text field
                        output_data = {'text': text}

                        # Write the new data to the output file
                        outfile.write(json.dumps(output_data) + '\n')
                    
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line: {line.strip()}")
                    continue
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue

        print(f"\nDetokenization complete. Output saved to '{output_path}'")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")

def main():
    parser = argparse.ArgumentParser(description="Detokenize the 'text' field of a JSONL file.")
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default paths relative to the script's location
    default_input = os.path.join(script_dir, "tokenized", "climblab", "climblab.jsonl")
    default_output = os.path.join(script_dir, "detokenized", "climblab", "climblab.jsonl")

    parser.add_argument('--input_path', type=str, default=default_input,
                        help=f"Path to the input JSONL file. Defaults to '{default_input}'")
    parser.add_argument('--output_path', type=str, default=default_output,
                        help=f"Path to save the output JSONL file. Defaults to '{default_output}'")
    
    args = parser.parse_args()
    
    detokenize_jsonl(args.input_path, args.output_path)

if __name__ == "__main__":
    main()
