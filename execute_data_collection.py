import yaml
import json
import google.generativeai as genai

# Configure the Gemini API
yaml_file_path = "demo_formular.yaml"
output_file_path = "output.yaml"
GEMINI_API_KEY = ""
genai.configure(api_key=GEMINI_API_KEY)


def read_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def write_yaml(data, file_path):
    with open(file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


#  Don't include "```yaml" at the start. If the input is empty string, return the starting document without "```yaml" at the start. <-- add this if you want flash gemini
def process_with_gemini(yaml_data, input_text):
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"""
    Given the following YAML structure:
    {yaml.dump(yaml_data)}
    
    And the following input text:
    {input_text}
    
    Please extract relevant information from the input text and update the YAML fields.
    Provide your response as a complete YAML document that includes all fields from the original structure, 
    updating only the relevant fields based on the input text.
    
    Ensure your response is a valid YAML document that can be parsed directly. Don't include ```yaml at start nor ``` at the end of your response.
    """
    response = model.generate_content(prompt)
    return response.text


def parse_gemini_response(response_text):
    try:
        # First, try to parse as YAML
        return yaml.safe_load(response_text)
    except yaml.YAMLError:
        try:
            # If YAML parsing fails, try JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            print("Failed to parse Gemini's response as YAML or JSON.")
            return None


def update_yaml(original_yaml, updated_yaml):
    """
    Recursively update the original YAML with values from the updated YAML,
    preserving the structure and any fields not present in the update.
    """
    if isinstance(updated_yaml, dict):
        for key, value in updated_yaml.items():
            if key in original_yaml:
                if isinstance(value, (dict, list)):
                    update_yaml(original_yaml[key], value)
                else:
                    original_yaml[key] = value
            else:
                original_yaml[key] = value
    elif isinstance(updated_yaml, list):
        for i, item in enumerate(updated_yaml):
            if i < len(original_yaml):
                if isinstance(item, (dict, list)):
                    update_yaml(original_yaml[i], item)
                else:
                    original_yaml[i] = item
            else:
                original_yaml.append(item)


# Get additional text input
def decode(input_text):
    # input_text = "male gender, so boy from Croatia, Zagreb. There he lives, but was born in Bosnia. His name is Marin and last name Jezidzic"
    yaml_data = read_yaml(yaml_file_path)
    # Process with Gemini
    gemini_response = process_with_gemini(yaml_data, input_text)
    # Parse Gemini's response
    updated_yaml = parse_gemini_response(gemini_response)
    print(updated_yaml)
    if updated_yaml:
        # Update the original YAML with the new information
        update_yaml(yaml_data, updated_yaml)

        # Write the updated YAML
        write_yaml(yaml_data, output_file_path)
        print(f"Updated YAML has been written to {output_file_path}")
    else:
        print("Failed to update YAML due to parsing error.")
