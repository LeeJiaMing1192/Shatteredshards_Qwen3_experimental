import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import psutil
import gc

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    gpu_usage = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024 / 1024  # MB
            reserved = torch.cuda.memory_reserved(i) / 1024 / 1024  # MB
            gpu_usage[f'GPU_{i}'] = {
                'allocated': allocated,
                'reserved': reserved
            }
    
    return ram_usage, gpu_usage

def print_memory_usage(stage, prev_ram, prev_gpu):
    """Print memory usage change for a given stage"""
    current_ram, current_gpu = get_memory_usage()
    
    print(f"\n=== Memory Usage at: {stage} ===")
    print(f"RAM: {current_ram:.2f} MB (Δ: {current_ram - prev_ram:+.2f} MB)")
    
    if current_gpu:
        for device, memory in current_gpu.items():
            prev_memory = prev_gpu.get(device, {'allocated': 0, 'reserved': 0})
            alloc_delta = memory['allocated'] - prev_memory['allocated']
            reserved_delta = memory['reserved'] - prev_memory['reserved']
            print(f"{device}: Allocated: {memory['allocated']:.2f} MB (Δ: {alloc_delta:+.2f} MB), "
                  f"Reserved: {memory['reserved']:.2f} MB (Δ: {reserved_delta:+.2f} MB)")
    
    return current_ram, current_gpu

def clear_memory():
    """Clear GPU and CPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def quick_shattered_test():
    """Quick test for the shattered model only with detailed memory tracking"""
    
    model_path = "qwen/Qwen3-4B-Thinking-2507"
    ##C:\\Users\\ADMIN\\Desktop\\4B_shattered_shards_code\\Qwen3-4B-thinking-2507-dynamic-kvC:\\Users\\ADMIN\\Desktop\\4B_shattered_shards_code\\Qwen3-4B-thinking-2507-dynamic-kv
    print(" Testing regular Qwen3-4B Model")
    print("=" * 50)
    
    # Initial memory
    initial_ram, initial_gpu = get_memory_usage()
    print(f" Initial RAM: {initial_ram:.2f} MB")
    if initial_gpu:
        for device, memory in initial_gpu.items():
            print(f" {device}: Allocated: {memory['allocated']:.2f} MB, Reserved: {memory['reserved']:.2f} MB")
    
    # Load tokenizer
    print("\n Loading tokenizer...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer_load_time = time.time() - start_time
    
    ram_after_tokenizer, gpu_after_tokenizer = print_memory_usage("After tokenizer", initial_ram, initial_gpu)
    print(f"⏱️ Tokenizer load time: {tokenizer_load_time:.2f}s")
    
    # Load model
    print("\n Loading model...")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model_load_time = time.time() - start_time
    total_load_time = tokenizer_load_time + model_load_time
    
    ram_after_model, gpu_after_model = print_memory_usage("After model load", ram_after_tokenizer, gpu_after_tokenizer)
    print(f"⏱️ Model load time: {model_load_time:.2f}s")
    print(f"⏱️ Total load time: {total_load_time:.2f}s")
    
    # Model info
    print(f"\n Model Class: {model.__class__.__name__}")
    print(f" Model Type: {model.config.model_type}")
    print(f" Device: {next(model.parameters()).device}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters()) / 1e9  # Billions
    print(f" Parameters: {total_params:.2f}B")
    
    # Test generation with detailed timing
    print("\n離 Testing Generation:")
    test_prompts = [
        """You are tasked with a complex, four-part assignment. You must act as an expert consultant, combining historical knowledge, technical understanding, and creative problem-solving. Part 1: The Historical/Technical Foundation (Information Retrieval & Synthesis) A. Research and Summarize: Conduct a brief, yet comprehensive, comparison between two major historical periods of urban planning: Baroque City Planning (e.g., Paris under Haussmann or the planning of Washington D.C.) and Garden City Movement (pioneered by Ebenezer Howard). Your summary must focus on three specific contrasting elements: Dominant Geometric Philosophy: (e.g., radial, grid, organic, etc.) Socio-Economic Goal: (What problem was the planning trying to solve?) Role of Transportation: (How was transit—or lack thereof—integrated into the design?) B. Define a Metric: Define the mathematical formula and units for "Urban Density Flux" (D flux ​ )—a hypothetical metric that measures the rate of change of population density within a specific 1 km 2 area over a period of one business day (8 hours), accounting for both residential population and estimated transient commercial/office population. Part 2: The Creative Application (Scenario Development & Constraint Management) Imagine you are planning a Neo-Renaissance district called "Aetheria" for a new, mid-sized coastal city. Aetheria will be 4 km 2 in size. Constraint Checklist (Must be strictly followed): A. The entire district must have at least 40% green space by area. B. No internal combustion engine vehicles are allowed inside the district's core 2 km 2 . C. The district's primary power source must be a combination of geothermal energy and passive solar architecture. D. The central hub must be a multi-use civic structure that is exactly 75 meters tall and includes both a library and an astronomical observatory. Develop a short, descriptive narrative (approx. 200 words) that details: How you architecturally harmonize the Baroque focus on grand axes/vistas with the Garden City's emphasis on green belts and community living. Your solution for the core transport issue (Constraint B). Part 3: The Structured Output (Data Generation & Formatting) Present the following information in a single, strictly-formatted Markdown Table: Planning Element	Baroque City Planning	Garden City Movement	Aetheria District Primary Shape	(Based on Part 1A)	(Based on Part 1A)	(Describe Aetheria's shape) Green Space %	(Estimate based on historical examples)	(Estimate based on historical examples)	(From Constraint A) Transport Model	(From Part 1A)	(From Part 1A)	(From Part 2) Max Building Height	(Suggest a typical/representative height in modern units)	(Suggest a typical/representative height in modern units)	(Height of the Central Hub in meters - from Constraint D) Export to Sheets Part 4: Logical Reasoning & Self-Correction Identify the single greatest potential long-term social or logistical vulnerability of the Aetheria design (Parts 2 & 3), specifically arising from the conflicting constraints (Baroque axes vs. green space, no cars vs. coastal city). Then, suggest one specific, creative, and cost-effective technological or policy adjustment to mitigate that vulnerability.""",
        
    ]
    
    generation_stats = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*40}")
        print(f"Test {i}: '{prompt}'")
        print(f"{'='*40}")
        
        # Clear cache before each test
        clear_memory()
        ram_before_gen, gpu_before_gen = get_memory_usage()
        
        # Encoding phase
        encode_start = time.time()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        encode_time = time.time() - encode_start
        
        # Generation phase
        gen_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2000,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_time = time.time() - gen_start
        
        # Decoding phase
        decode_start = time.time()
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        decode_time = time.time() - decode_start
        
        total_time = encode_time + gen_time + decode_time
        
        # Memory after generation
        ram_after_gen, gpu_after_gen = get_memory_usage()
        
        # Calculate tokens per second
        input_tokens = inputs.input_ids.shape[1]
        output_tokens = outputs.shape[1] - input_tokens
        tokens_per_sec = output_tokens / gen_time if gen_time > 0 else 0
        
        print(f" Input tokens: {input_tokens}")
        print(f" Output tokens: {output_tokens}")
        print(f" Total tokens: {input_tokens + output_tokens}")
        print(f"⏱️ Encoding time: {encode_time:.3f}s")
        print(f"⏱️ Generation time: {gen_time:.3f}s")
        print(f"⏱️ Decoding time: {decode_time:.3f}s")
        print(f"⏱️ Total time: {total_time:.3f}s")
        print(f"⚡ Tokens per second: {tokens_per_sec:.2f}")
        print(f" Response: {response[:150]}...")
        
        # Memory deltas
        print(f" RAM Δ: {ram_after_gen - ram_before_gen:+.2f} MB")
        if gpu_after_gen:
            for device, memory in gpu_after_gen.items():
                prev_memory = gpu_before_gen.get(device, {'allocated': 0, 'reserved': 0})
                alloc_delta = memory['allocated'] - prev_memory['allocated']
                print(f" {device} Allocated Δ: {alloc_delta:+.2f} MB")
        
        # Store stats
        generation_stats.append({
            'test': i,
            'prompt': prompt,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'encode_time': encode_time,
            'gen_time': gen_time,
            'decode_time': decode_time,
            'total_time': total_time,
            'tokens_per_sec': tokens_per_sec
        })
    
    # Print summary statistics
    print("\n" + "="*60)
    print(" GENERATION PERFORMANCE SUMMARY")
    print("="*60)
    
    avg_tokens_per_sec = sum(stat['tokens_per_sec'] for stat in generation_stats) / len(generation_stats)
    avg_gen_time = sum(stat['gen_time'] for stat in generation_stats) / len(generation_stats)
    avg_total_time = sum(stat['total_time'] for stat in generation_stats) / len(generation_stats)
    
    print(f" Average tokens per second: {avg_tokens_per_sec:.2f}")
    print(f" Average generation time: {avg_gen_time:.3f}s")
    print(f" Average total time: {avg_total_time:.3f}s")
    print(f" Total tests: {len(generation_stats)}")
    
    # Final memory usage
    print("\n" + "="*40)
    print(" FINAL MEMORY USAGE")
    print("="*40)
    final_ram, final_gpu = get_memory_usage()
    print(f" Final RAM: {final_ram:.2f} MB")
    if final_gpu:
        for device, memory in final_gpu.items():
            print(f" {device}: Allocated: {memory['allocated']:.2f} MB, Reserved: {memory['reserved']:.2f} MB")
    
    # Clean up
    print("\n粒 Cleaning up...")
    del model
    del tokenizer
    clear_memory()
    
    # Memory after cleanup
    ram_after_cleanup, gpu_after_cleanup = get_memory_usage()
    print(f" RAM after cleanup: {ram_after_cleanup:.2f} MB (Δ: {ram_after_cleanup - final_ram:+.2f} MB)")
    if gpu_after_cleanup:
        for device, memory in gpu_after_cleanup.items():
            prev_memory = final_gpu.get(device, {'allocated': 0, 'reserved': 0})
            alloc_delta = memory['allocated'] - prev_memory['allocated']
            print(f" {device} Allocated after cleanup: {memory['allocated']:.2f} MB (Δ: {alloc_delta:+.2f} MB)")

if __name__ == "__main__":
    quick_shattered_test()