import json
import csv
import re

def robust_parse_file(filepath):
    """
    Reads the input text file and attempts to parse its content as a JSON array of dictionaries.
    If the content is not wrapped as a proper JSON array, it adds the necessary brackets.
    Lines that begin with ellipses (e.g., "...") are filtered out. In the event JSON parsing fails,
    a regex fallback is used to extract JSON-like objects.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Filter out lines that are just ellipses or non-data markers.
    lines = content.splitlines()
    filtered_lines = [line for line in lines if not line.strip().startswith("...")]
    content = "\n".join(filtered_lines).strip()
    
    # If the content doesn't start and end with square brackets, wrap it to form a JSON array.
    if not (content.startswith('[') and content.endswith(']')):
        content = "[" + content.rstrip(',') + "]"
    
    try:
        data = json.loads(content)
    except Exception as e:
        print(f"JSON load failed with error: {e}")
        # Fallback: Use regex to extract portions that look like JSON objects.
        matches = re.findall(r'\{.*?\}', content, re.DOTALL)
        data = []
        for match in matches:
            try:
                entry = json.loads(match)
                data.append(entry)
            except Exception as inner_e:
                print(f"Skipping an entry due to error: {inner_e}")
    return data

def write_csv(data, output_filepath):
    """
    Writes the parsed data (a list of dictionaries) into a CSV file with columns 'text' and 'labels'.
    The csv module automatically handles quoting to manage any commas within the text.
    """
    fieldnames = ['text', 'labels']
    with open(output_filepath, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for entry in data:
            writer.writerow({
                'text': entry.get("text", ""),
                'labels': entry.get("labels", "")
            })

if __name__ == "__main__":
    input_file = "reasoning_examples.txt"
    output_file = "reasoning_examples.csv"
    parsed_data = robust_parse_file(input_file)
    if parsed_data:
        write_csv(parsed_data, output_file)
        print(f"CSV file successfully written to {output_file}")
    else:
        print("No valid data found to write.") 