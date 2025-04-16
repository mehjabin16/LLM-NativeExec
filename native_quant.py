# LLM Benchmarking Script - Python with Hugging Face (Quantized Version)

import os
import json
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

login(token = 'your_key')

def read_data_file(file_path):
    """Read JSON or JSONL data files."""
    print(f"Loading data from: {file_path}")
    
    with open(file_path, 'r') as f:
        if file_path.endswith('.jsonl'):
            data = []
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
            return data
        else:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON data is not an array")
            return data

def format_prompt(instruction, input_text="", model_name=""):
    """Format the prompt based on model type."""
    if "llama" in model_name.lower():
        # Llama formatting
        if input_text and input_text.strip():
            prompt = f"<s>[INST] {instruction}\n\n{input_text} [/INST]"
        else:
            prompt = f"<s>[INST] {instruction} [/INST]"
    elif "phi" in model_name.lower():
        # Phi formatting
        prompt = f"Instruct: {instruction}\n"
        if input_text and input_text.strip():
            prompt += f"Input: {input_text}\n"
        prompt += "Output: "
    elif "gemma" in model_name.lower():
        # Gemma formatting
        if input_text and input_text.strip():
            prompt = f"<start_of_turn>user\n{instruction}\n\n{input_text}<end_of_turn>\n<start_of_turn>model\n"
        else:
            prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
    elif "qwen" in model_name.lower():
        # Qwen formatting
        if input_text and input_text.strip():
            prompt = f"<|im_start|>user\n{instruction}\n\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    elif "tinyllama" in model_name.lower() or "chat" in model_name.lower():
        # TinyLlama and other chat models
        if input_text and input_text.strip():
            prompt = f"<|user|>\n{instruction}\n\n{input_text}\n<|assistant|>\n"
        else:
            prompt = f"<|user|>\n{instruction}\n<|assistant|>\n"
    else:
        # Generic prompt format for other models
        prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\nInstruction:\n{instruction}\n\n"
        if input_text and input_text.strip():
            prompt += f"Input:\n{input_text}\n\n"
        prompt += "Response:"

    return prompt

class Recorder:
    """Track timing metrics across multiple operations."""
    def __init__(self):
        self.time = {}
        self.cnt = {}

    def record(self, name, num):
        if name not in self.time:
            self.time[name] = 0.0
            self.cnt[name] = 0
        self.time[name] += num
        self.cnt[name] += 1

    def get(self, name):
        return self.time[name], self.cnt[name]

class Timer:
    """Context manager for timing operations with CUDA synchronization."""
    def __init__(self, name, recorder):
        self.name = name
        self.recorder = recorder
        self._start_time = None
        self._end_time = None

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()

    def __exit__(self, *exc_info):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._end_time = time.perf_counter()
        elapsed_time = self._end_time - self._start_time
        self.recorder.record(self.name, elapsed_time * 1000)  # Convert to ms

def generate_with_timing(model, tokenizer, prompt, device, recorder):
    """Generate text with timing metrics for prefill and decode phases."""
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    input_token_count = input_ids.shape[1]
    
    all_output_tokens = []
    completion_token_count = 0
    
    with Timer("prefill", recorder):
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                return_dict_in_generate=True,
                output_scores=True
            )
            
    first_new_token = outputs.sequences[0, -1].unsqueeze(0)
    all_output_tokens.append(first_new_token)
    completion_token_count += 1
    
    # Current sequence with first generated token
    current_input_ids = outputs.sequences
    
    # Track token-by-token decoding time
    decode_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    # Continue generating tokens (decode phase)
    for _ in range(511):  # Max 512 new tokens (including first token)
        with Timer("decode", recorder):
            with torch.no_grad():
                outputs = model.generate(
                    current_input_ids,
                    max_new_tokens=1,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    return_dict_in_generate=True
                )
        
        # Get the new token
        new_token = outputs.sequences[0, -1].unsqueeze(0)
        completion_token_count += 1
        all_output_tokens.append(new_token)
        
        # Update input for next iteration
        current_input_ids = outputs.sequences
        
        # Check if we've generated an EOS token
        if tokenizer.eos_token_id is not None and new_token.item() == tokenizer.eos_token_id:
            break
            
    # Calculate total time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_end = time.time()
    decode_time = (decode_end - decode_start) * 1000  # ms
    
    # Stack all tokens and decode
    if all_output_tokens:
        all_tokens = torch.cat(all_output_tokens, dim=0)
        generated_text = tokenizer.decode(all_tokens, skip_special_tokens=True)
    else:
        generated_text = ""
    
    # Get timing metrics from recorder
    prefill_time, _ = recorder.get("prefill")
    decode_time_sum, decode_count = recorder.get("decode")
    
    # Calculate speeds
    prefill_tokens_per_s = input_token_count / (prefill_time / 1000) if prefill_time > 0 else 0
    decode_tokens_per_s = decode_count / (decode_time_sum / 1000) if decode_time_sum > 0 else 0
    
    # Total time
    total_inference_time = prefill_time + decode_time_sum
    
    # Build usage info
    usage = {
        "prompt_tokens": input_token_count,
        "completion_tokens": completion_token_count,
        "total_tokens": input_token_count + completion_token_count,
        "extra": {
            "prefill_tokens_per_s": prefill_tokens_per_s,
            "decode_tokens_per_s": decode_tokens_per_s,
        }
    }
    
    # Calculate normalized latency
    normalized_latency = total_inference_time / completion_token_count if completion_token_count > 0 else 0
    
    return {
        "output": generated_text,
        "total_time": total_inference_time,
        "time_to_first_token": prefill_time,
        "decode_time": decode_time_sum,
        "usage": usage,
        "normalized_latency": normalized_latency
    }

def load_model_and_tokenizer(model_name, device="auto", quantization="none"):
    """Load model and tokenizer with appropriate settings and quantization."""
    print(f"Loading model: {model_name} with quantization: {quantization}")
    
    # Determine device 
    if device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device(device)
        
    # Load tokenizer first
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None, None
    
    try:
        # Set up quantization configuration
        start_time = time.time()
        model_kwargs = {"device_map": device if device != "mps" else "cpu", "low_cpu_mem_usage": True}
        
        if quantization == "none" or device == "mps":  # MPS doesn't support quantization
            model_kwargs["torch_dtype"] = torch.float16
        elif quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        elif quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move to MPS specifically if using Apple Silicon
        if str(device) == "mps":
            model = model.to(device)
            
        load_time = (time.time() - start_time) * 1000  # ms
        print(f"Model loaded in {load_time:.2f} ms")
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def process_file_with_model(model_name, model, tokenizer, data_file, device):
    """Process a single JSON file with one model."""
    try:
       
        data = read_data_file(data_file)
        model_results = []
        recorder = Recorder()
        
        
        for i, sample in enumerate(tqdm(data, desc=f"Processing {os.path.basename(data_file)}")):
            instruction = sample.get('instruction', '')
            input_text = sample.get('input', '')
            
            prompt = format_prompt(instruction, input_text, model_name)
            
            result = generate_with_timing(model, tokenizer, prompt, device, recorder)
            
            model_results.append({
                "instruction": instruction,
                "input": input_text,
                "output": result["output"],
                "Prefill speed": f"{result['usage']['extra']['prefill_tokens_per_s']:.2f}",
                "Decoding speed": f"{result['usage']['extra']['decode_tokens_per_s']:.2f}",
                "Total inference time": f"{result['total_time']:.2f} ms",
                "Time to first token": f"{result['time_to_first_token']:.2f} ms",
            })
        
       
        prefill_time, prefill_count = recorder.get("prefill")
        decode_time, decode_count = recorder.get("decode")
        
        
        avg_prefill_time = prefill_time / prefill_count if prefill_count else 0
        avg_decode_time = decode_time / decode_count if decode_count else 0
        avg_prefill_speed = sum([float(r["Prefill speed"].split()[0]) for r in model_results]) / len(model_results)
        avg_decode_speed = sum([float(r["Decoding speed"].split()[0]) for r in model_results]) / len(model_results)
        
        # Calculate normalized latency (ms/token)
        total_tokens_generated = sum([r["usage"]["completion_tokens"] for r in model_results])
        total_time = prefill_time + decode_time
        latency_metric = total_time / total_tokens_generated if total_tokens_generated > 0 else 0
        
   
        summary = {
            "Average Prefill Time": f"{avg_prefill_time:.2f} ms",
            "Average Decode Time per Token": f"{avg_decode_time:.2f} ms",
            "Average Prefill Speed": f"{avg_prefill_speed:.2f} tokens/sec",
            "Average Decoding Speed": f"{avg_decode_speed:.2f} tokens/sec",
            "Latency Metric (Total Time / Total Tokens)": f"{latency_metric:.2f} ms/token",
        }
        
        model_results.append(summary)
        
        print("\n" + "="*50)
        print(f"Results for {model_name} on {os.path.basename(data_file)}:")
        for metric, value in summary.items():
            print(f"{metric}: {value}")
        print("="*50 + "\n")
        
        return model_results
        
    except Exception as e:
        print(f"Error processing file {data_file} with model {model_name}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmarks using Hugging Face models with quantization")
    parser.add_argument("--models", nargs="+", 
                        help="List of model names to benchmark (overrides default list)")
    parser.add_argument("--json_files", nargs="+", required=True,
                        help="List of JSON files to process")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to run on: 'cuda', 'cpu', 'mps', or 'auto'")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--quantization", type=str, choices=["none", "4bit", "8bit"], default="none",
                        help="Quantization mode: none, 4bit, or 8bit")
    args = parser.parse_args()
    
    default_models = [
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "microsoft/phi-2", 
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "google/gemma-2-2b-it"
    ]
    
    models_to_benchmark = args.models if args.models else default_models
    
    os.makedirs(args.output_dir, exist_ok=True)
    for model_name in models_to_benchmark:
        print(f"\nBenchmarking model: {model_name}")
        

        model, tokenizer = load_model_and_tokenizer(
            model_name, 
            device=args.device,
            quantization=args.quantization
        )
        
        if model is None or tokenizer is None:
            print(f"Skipping model {model_name} due to loading errors")
            continue
            
     
        for json_file in args.json_files:
            if not os.path.exists(json_file):
                print(f"Warning: File {json_file} not found. Skipping.")
                continue
            
            file_basename = os.path.basename(json_file)
            file_number = "001"
            if match := file_basename.split('.')[0].split('_'):
                if len(match) > 1 and match[1].isdigit():
                    file_number = match[1]
          
            model_name_clean = model_name.replace("/", "-")
            if "/" in model_name:
                model_name_clean = model_name.split("/")[-1]
                
            results = process_file_with_model(model_name, model, tokenizer, json_file, model.device)
            
            quant_suffix = f"_{args.quantization}" if args.quantization != "none" else ""
            output_file = os.path.join(args.output_dir, f"{model_name_clean}{quant_suffix}_{file_number}_results.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {output_file}")
            
        # Clear model from memory
        del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"Finished benchmarking {model_name}\n")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 
    main()