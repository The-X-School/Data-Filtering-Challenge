import json
import os
import sys

def format_glaive_dataset(input_file, output_dir, output_filename):
    """
    Process the Glaive function calling dataset and convert it to LMFlow format.
    
    Args:
        input_file: Path to the input JSONL file
        output_dir: Directory to create for the output
        output_filename: Name of the output JSON file
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Processing Glaive dataset from: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Output file: {output_path}")
    
    instances = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        line_count += 1
                        
                        # Extract the system and chat fields
                        system_prompt = record.get('system', '')
                        chat_content = record.get('chat', '')
                        
                        # Combine system prompt and chat into a single text field
                        # Format it in a way that's useful for training
                        if system_prompt and chat_content:
                            combined_text = f"System: {system_prompt}\n\n{chat_content}"
                        elif chat_content:
                            combined_text = chat_content
                        elif system_prompt:
                            combined_text = system_prompt
                        else:
                            continue  # Skip empty records
                        
                        # Add to instances in LMFlow format
                        instances.append({
                            "text": combined_text
                        })
                        
                        # Progress feedback
                        if line_count % 10000 == 0:
                            print(f"Processed {line_count:,} records...")
                            
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_count + 1}: {e}")
                        continue
        
        print(f"Successfully processed {len(instances):,} records.")
        
        # Create the LMFlow format structure
        output_data = {
            "type": "text_only",
            "instances": instances
        }
        
        # Write to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Successfully created filtered dataset: {output_path}")
        print(f"📊 Total instances: {len(instances):,}")
        
        # Show a sample of the first record for verification
        if instances:
            sample_text = instances[0]["text"][:200] + "..." if len(instances[0]["text"]) > 200 else instances[0]["text"]
            print(f"📝 Sample text: {sample_text}")
            
    except FileNotFoundError:
        print(f"❌ Error: Input file not found: {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        sys.exit(1)

def main():
    # Default paths
    input_file = os.path.join("data", "glaive", "glaive.jsonl")
    output_dir = os.path.join("data", "glaive_filtered")
    output_filename = "glaive_filtered.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("Please make sure you have downloaded the Glaive dataset first using glaive_download.py")
        sys.exit(1)
    
    # Process the dataset
    format_glaive_dataset(input_file, output_dir, output_filename)

if __name__ == "__main__":
    main() 