import json
import time
import argparse
import os
from tqdm import tqdm
from mlc_llm import MLCEngine

# Function to load the input data (handles both JSON and JSONL)
def read_data_file(file_path):
    """Read JSON or JSONL data files."""
    print(f"Loading data from: {file_path}")
    with open(file_path, 'r') as f:
        if file_path.endswith('.jsonl'):
            # Process as JSON Lines file
            data = []
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
            return data
        else:
            # Process as regular JSON file
            data = json.load(f)
            if isinstance(data, dict):
                # Convert single entry to list
                return [data]
            elif isinstance(data, list):
                return data
            else:
                raise ValueError("JSON data format not recognized")

# Format prompt based on model requirements
def format_prompt(instruction, input_text):
    """Format the prompt for the model."""
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    elif instruction:
        return instruction
    else:
        return input_text

def generate_with_mlc(model_path, entry):
    """Generate output using MLC LLM and collect performance metrics from MLC's native metrics."""
    # Initialize the MLC engine
    engine = MLCEngine(model_path)
    
    instruction = entry.get('instruction', '')
    input_text = entry.get('input', '')
    
    # Format prompt
    prompt = format_prompt(instruction, input_text)
    
    # Prepare messages for the LLM
    messages = [{"role": "user", "content": prompt}]
    
    # Storage for output tokens
    output_tokens = []
    
    # Start time to measure inference performance
    start_time = time.time()
    
    # Generate response with streaming
    for response in engine.chat.completions.create(
        messages=messages,
        model=model_path,
        stream=True,
    ):
        for choice in response.choices:
            token = choice.delta.content
            output_tokens.append(token)
            print(token, end="", flush=True)
    
    # Get metrics directly from the engine
    raw_metrics = engine.get_metrics()  # This is the key addition - assumes an API exists
    
    # If the above method doesn't exist, you could try accessing metrics via a method similar to CLI's /metrics
    # raw_metrics = engine.metrics()  # Alternative method name
    
    # For reference, metrics we expect to find (based on your /metrics example):
    # - engine_prefill_time_sum
    # - engine_decode_time_sum 
    # - engine_jump_forward_time_sum
    # - prompt_tokens_sum
    # - completion_tokens_sum
    # - prefill_tokens_sum
    # - decode_tokens_sum
    # - jump_forward_tokens_sum
    # - prefill_tokens_per_s
    
    # Format the raw metrics into our desired structure
    prefill_time = raw_metrics.get('engine_prefill_time_sum', 0)
    decode_time = raw_metrics.get('engine_decode_time_sum', 0)
    prompt_tokens = raw_metrics.get('prompt_tokens_sum', 0)
    completion_tokens = raw_metrics.get('completion_tokens_sum', 0)
    prefill_tokens_per_s = raw_metrics.get('prefill_tokens_per_s', 0)
    
    # Calculate additional metrics not provided directly
    total_inference_time = (time.time() - start_time) * 1000  # in ms
    total_tokens = prompt_tokens + completion_tokens
    
    # Join tokens for final output
    output_text = "".join(output_tokens)
    
    # Create usage info structure consistent with WebLLM
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "extra": {
            "prefill_tokens_per_s": prefill_tokens_per_s,
            "decode_tokens_per_s": completion_tokens / decode_time if decode_time > 0 else 0,
        }
    }
    
    # Create output data incorporating MLC's metrics
    output_data = {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "Prefill speed": f"{prefill_tokens_per_s:.2f}",
        "Decoding speed": f"{completion_tokens / decode_time:.2f}" if decode_time > 0 else "N/A",
        "Total inference time": f"{total_inference_time:.2f} ms",
        "metrics": {
            "total_inference_time": total_inference_time,
            "tokens_generated": completion_tokens,
            "raw_mlc_metrics": raw_metrics,  # Store all raw metrics for reference
            "usage": usage
        }
    }
    
    # Terminate the engine to clean up
    engine.terminate()
    
    return output_data

# Main function to handle command-line arguments and run the benchmark
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run LLM Benchmarking with MLC")
    parser.add_argument("--input_dir", type=str, default=".", help="Directory containing input JSON files")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output JSON files")
    parser.add_argument("--specific_model", type=str, help="Run only a specific model from the default list")
    parser.add_argument("--specific_file", type=str, help="Run only a specific file from the default list")
    args = parser.parse_args()
    
    # Define default models
    default_models = [
        "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC"
    ]
    
    # Define default JSON files
    json_files = [
        'alpaca_001.json'
    ]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Filter models if specific model is provided
    models_to_run = [args.specific_model] if args.specific_model else default_models
    
    # Filter files if specific file is provided
    files_to_run = [args.specific_file] if args.specific_file else json_files
    files_to_run = [os.path.join(args.input_dir, f) for f in files_to_run]
    
    # Run benchmarks for all selected models and files
    for model in models_to_run:
        print(f"\n=== Running benchmarks for model: {model} ===")
        
        # Clean model name for file naming
        model_name_clean = model.replace('/', '_').replace('-', '_').replace(':', '_')
        
        for json_file in files_to_run:
            print(f"\nProcessing file: {json_file}")
            
            # Extract file name for output naming
            file_name = os.path.basename(json_file).split('.')[0]
            
            try:
                # Load input data
                input_data = read_data_file(json_file)
                
                # Process each entry
                all_results = []
                
                for idx, entry in enumerate(tqdm(input_data, desc="Processing entries")):
                    print(f"\nProcessing entry {idx+1}/{len(input_data)}")
                    
                    # Run the model inference
                    result = generate_with_mlc(model, entry)
                    all_results.append(result)
                    
                    print("\n---")
                
                # Calculate aggregate metrics
                total_time = sum(r["metrics"]["total_inference_time"] for r in all_results)
                total_tokens = sum(r["metrics"]["tokens_generated"] for r in all_results)
                
                aggregate_metrics = {
                    "total_samples": len(all_results),
                    "total_inference_time": total_time,
                    "total_tokens_generated": total_tokens,
                    "average_time_per_sample": total_time / len(all_results) if all_results else 0,
                    "average_tokens_per_sample": total_tokens / len(all_results) if all_results else 0,
                    "overall_throughput_tokens_per_sec": total_tokens / total_time if total_time > 0 else 0
                }
                
                # Save detailed results
                output_file = os.path.join(args.output_dir, f"{model_name_clean}_{file_name}_results.json")
                output_data = {
                    "model": model,
                    "input_file": json_file,
                    "detailed_results": all_results,
                    "aggregate_metrics": aggregate_metrics
                }
                
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"Results saved to: {output_file}")
                
            except Exception as e:
                print(f"Error processing {json_file} with model {model}: {str(e)}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()