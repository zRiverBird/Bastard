import argparse
import json
import os
import time
from typing import Any, Dict, List
from tqdm import tqdm
from openai import OpenAI
from pprint import pprint as pp
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description='VLM Multi-GPU Inference using OpenAI API')
    parser.add_argument('--model', type=str, required=True, help='VLMs')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input data JSON file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output JSON file')
    parser.add_argument('--max_model_len', type=int, default=8192,
                        help='Maximum model length')
    # hyperparameters
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.2,
                        help='Top-p for sampling')
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1",
                        help='vLLM API base URL')
    return parser.parse_args()



class VLMAPIInference:
    def __init__(self, model_name: str, api_base: str, temperature: float, 
                 top_p: float, max_tokens: int):
        print(api_base)
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=api_base
        )
        self.model = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        self.system = "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
        self.question = "This is a picture of an autonomous driving scene. Based on the provided image, the road scene around the vehicle is described in detail." # TODO

    def process_sample(self, key: str, img_paths: Dict[str, str]) -> str:
        res = {}
        for camera_view, img_path in img_paths.items():
            messages = [
                {"role": "system", "content": self.system}
            ]
            content = []
            img_path = os.path.join("/aojidata-sh/llm/Qwen2.5-VL/qwen-vl-finetune/data/16g/nuscenes", img_path)
            
            try:
                if any(img_path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.webm']):
                    print(f"Warning: Video input detected: {img_path}. Video processing is disabled.")
                    continue
                
                if not os.path.exists(img_path):
                    print(f"Warning: Image file not found: {img_path}. Skipping.")
                    continue
                    
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"file://{os.path.abspath(img_path)}"
                    }
                })
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}. Skipping.")
                continue
        
            if not content:
                return "Error: No valid images found to process."
            
            # Add the question
            content.append({
                "type": "text",
                "text": self.question
            })
            
            messages.append({
                "role": "user",
                "content": content
            })

            # test
            # breakpoint()
            # pp(messages)
            # time.sleep(1)
            res[camera_view] = "res..."

            # client
            # try:
            #     # Call the API with retries
            #     max_retries = 3
            #     retry_delay = 1
                
            #     for attempt in range(max_retries):
            #         try:
            #             response = self.client.chat.completions.create(
            #                 model=self.model,
            #                 messages=messages,
            #                 temperature=self.temperature,
            #                 top_p=self.top_p,
            #                 max_tokens=self.max_tokens
            #             )
            #             res[camera_view] = response.choices[0].message.content
            #             break
            #             # return response.choices[0].message.content
            #         except Exception as e:
            #             if attempt == max_retries - 1:
            #                 raise 
            #             print(f"API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}. Retrying in {retry_delay}s...")
            #             time.sleep(retry_delay)
            #             retry_delay *= 2
                        
            # except Exception as e:
            #     error_msg = f"Error calling API: {str(e)}"
            #     print(error_msg)
            #     return error_msg
        
        return res

def load_or_create_output(output_path: str) -> List[Dict[str, Any]]:
    """Load existing output if it exists, or create a new output file."""
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
            print(f"Loaded {len(existing_data)} existing results from {output_path}")
            return existing_data
        except Exception as e:
            print(f"Error loading existing output file: {str(e)}. Starting fresh.")
            return []
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return []

def save_output(output_path: str, data: List[Dict[str, Any]]):
    """Save output data to file."""
    temp_path = output_path + '.tmp'
    try:
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, output_path)
    except Exception as e:
        print(f"Error saving output: {str(e)}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

def process_qa_data(vlm: VLMAPIInference, data: List[Dict[str, Any]], output_path: str) -> List[Dict[str, Any]]:
    """Process QA data and generate answers, saving results in real-time."""
    # Load existing results or create new output file
    output_data = load_or_create_output(output_path)
    
    # Process each remaining sample
    with tqdm(total=len(data), desc="Processing samples") as pbar:
        for k, v in data.items():
            output_v = v.copy()
            
            answer = vlm.process_sample(k, v["image_paths"])
            
            output_v['answer'] = answer
            
            output_data.append(output_v)
            
            save_output(output_path, output_data)
            
            pbar.update(1)
    
    return output_data

def main():
    args = parse_arguments()

    # Initialize VLM API client
    vlm = VLMAPIInference(
        model_name=args.model,
        api_base=args.api_base,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )

    # Load input data
    print(f"Loading input data from {args.data}")
    with open(args.data, 'r') as f:
        data = json.load(f)

    # Process data and generate answers
    print("Processing data and generating answers...")
    output_data = process_qa_data(vlm, data, args.output)

    print("Done!")

if __name__ == '__main__':
    main() 
