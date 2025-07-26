import json

def reformat_data(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            try:
                data = json.loads(line)

                if 'query' not in data or 'tools' not in data or 'answers' not in data:
                    continue

                query = data['query']
                
                # The 'tools' and 'answers' are JSON strings, so they need to be parsed
                tools_data = json.loads(data['tools'])
                answers_data = json.loads(data['answers'])
                
                # Let's create the tool definitions string
                tool_definitions = ""
                for tool in tools_data:
                    tool_definitions += f"Given the following function: {tool.get('name')} with description: {tool.get('description')} and parameters: {json.dumps(tool.get('parameters'))}. "

                # Let's create the question string
                question_str = f"Question: {tool_definitions}if you are asked to {query}, you will call..."

                # Let's get the last tool call from answers as the correct answer
                if answers_data:
                    last_answer = answers_data[-1]
                    # Reconstructing the tool call format from name and arguments
                    answer_tool_name = last_answer.get('name')
                    answer_arguments = last_answer.get('arguments', {})
                    args_str = ', '.join([f'{k}={repr(v)}' for k, v in answer_arguments.items()])
                    correct_answer = f"{answer_tool_name}({args_str})"
                    
                    answer_str = f"Answer: {correct_answer}"

                    # Combine into the final text format
                    final_text = f"{question_str}\n\n{answer_str}"

                    reformatted_entry = {
                        "text": final_text
                    }
                    outfile.write(json.dumps(reformatted_entry) + '\n')

            except json.JSONDecodeError:
                # Handle cases where a line is not a valid JSON
                print(f"Skipping malformed line: {line.strip()}")
                continue
            except Exception as e:
                print(f"An error occurred: {e}")
                continue


if __name__ == '__main__':
    input_dataset = 'Data-Filtering-Challenge/data/60kxlam/60kxlam.jsonl'
    output_dataset = 'Data-Filtering-Challenge/data/60kxlam_filtered.jsonl'
    reformat_data(input_dataset, output_dataset)
    print(f"Reformatted data saved to {output_dataset}") 