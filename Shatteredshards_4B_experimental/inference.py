# import torch
# from transformers import AutoTokenizer
# import time
# from architecture_implementation.lol import load_dynamic_model
# import psutil
# import os

# def get_memory_usage():
#     """Get current memory usage in GB"""
#     process = psutil.Process(os.getpid())
#     return process.memory_info().rss / 1024 / 1024 / 1024  # GB

# def get_gpu_memory():
#     """Get GPU memory usage in GB"""
#     if torch.cuda.is_available():
#         return torch.cuda.memory_allocated() / 1024 / 1024 / 1024
#     return 0

# def get_gpu_memory_reserved():
#     """Get GPU memory reserved in GB"""
#     if torch.cuda.is_available():
#         return torch.cuda.memory_reserved() / 1024 / 1024 / 1024
#     return 0

# def get_gpu_memory_cached():
#     """Get GPU memory cached in GB"""
#     if torch.cuda.is_available():
#         return torch.cuda.memory_cached() / 1024 / 1024 / 1024
#     return 0

# def print_memory_status(step_name, start_time=None, start_cpu=None, start_gpu=None, start_gpu_reserved=None):
#     """Print comprehensive memory status"""
#     current_cpu = get_memory_usage()
#     current_gpu = get_gpu_memory()
#     current_gpu_reserved = get_gpu_memory_reserved()
#     current_gpu_cached = get_gpu_memory_cached()
    
#     if start_time and start_cpu is not None and start_gpu is not None:
#         time_elapsed = time.time() - start_time
#         cpu_delta = current_cpu - start_cpu
#         gpu_delta = current_gpu - start_gpu
#         gpu_reserved_delta = current_gpu_reserved - start_gpu_reserved
        
#         print(f" {step_name}: {time_elapsed:.2f}s")
#         print(f"   CPU: {current_cpu:.2f}GB (Δ{cpu_delta:+.2f}GB)")
#         print(f"   GPU Allocated: {current_gpu:.2f}GB (Δ{gpu_delta:+.2f}GB)")
#         print(f"   GPU Reserved: {current_gpu_reserved:.2f}GB (Δ{gpu_reserved_delta:+.2f}GB)")
#         print(f"   GPU Cached: {current_gpu_cached:.2f}GB")
#     else:
#         print(f" {step_name}:")
#         print(f"   CPU: {current_cpu:.2f}GB")
#         print(f"   GPU Allocated: {current_gpu:.2f}GB")
#         print(f"   GPU Reserved: {current_gpu_reserved:.2f}GB")
#         print(f"   GPU Cached: {current_gpu_cached:.2f}GB")
    
#     return time.time(), current_cpu, current_gpu, current_gpu_reserved

# def dynamic_model_inference(model_path, prompt, max_length=3000):
#     """Run single prompt inference with dynamic KV cache model"""
#     print("=" * 60)
#     print("Dynamic Model Inference Test")
#     print("=" * 60)
    
#     # Initial memory state
#     start_time, start_cpu, start_gpu, start_gpu_reserved = print_memory_status("Initial state")
    
#     # Load model and tokenizer
#     print("\n Loading model and tokenizer...")
#     load_start = time.time()
    
#     model = load_dynamic_model(model_path)
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
    
#     load_time = time.time() - load_start
#     print(f"✅ Model loaded on device: {model.device}")
#     print(f"⏱️  Loading time: {load_time:.2f}s")
    
#     # Memory after loading
#     load_end_time, load_end_cpu, load_end_gpu, load_end_gpu_reserved = print_memory_status(
#         "After model loading", load_start, start_cpu, start_gpu, start_gpu_reserved
#     )
    
#     # Check device placement
#     print("\n Model Device Placement:")
#     total_params = 0
#     gpu_params = 0
#     cpu_params = 0
    
#     for name, param in model.named_parameters():
#         total_params += param.numel()
#         if param.device.type == 'cuda':
#             gpu_params += param.numel()
#         elif param.device.type == 'cpu':
#             cpu_params += param.numel()
    
#     print(f"   Total parameters: {total_params:,}")
#     print(f"   GPU parameters: {gpu_params:,} ({gpu_params/total_params*100:.1f}%)")
#     print(f"   CPU parameters: {cpu_params:,} ({cpu_params/total_params*100:.1f}%)")
    
#     # Encode input USING CHAT TEMPLATE (same as original model test)
#     print(f"\n️  Preparing input with Chat Template...")
    
#     # Use the EXACT same chat template approach as original model
#     messages = [
#         {"role": "user", "content": prompt}
#     ]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
#     print(f" Input prompt: {prompt}")
#     print(f" Input length: {len(model_inputs.input_ids[0])} tokens")
#     print(f" Max new tokens: {max_length}")
    
#     # Memory before generation
#     gen_start_time, gen_start_cpu, gen_start_gpu, gen_start_gpu_reserved = print_memory_status("Before generation")
    
#     # Generate with dynamic KV cache
#     print(f"\n⚡ Generating text with Dynamic KV Cache...")
    
#     with torch.no_grad():
#         generated_ids = model.generate(
#             **model_inputs,
#             max_new_tokens=max_length,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9,
#             repetition_penalty=1.1
#         )
    
#     generation_time = time.time() - gen_start_time
    
#     # Memory after generation
#     gen_end_time, gen_end_cpu, gen_end_gpu, gen_end_gpu_reserved = print_memory_status(
#         "After generation", gen_start_time, gen_start_cpu, gen_start_gpu, gen_start_gpu_reserved
#     )
    
#     # Parse thinking content (same as original model test)
#     print(f"\n Parsing output...")
#     output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

#     try:
#         # rindex finding 151668 (</think>)
#         index = len(output_ids) - output_ids[::-1].index(151668)
#     except ValueError:
#         index = 0

#     thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
#     content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    
#     print(f"\n易 THINKING CONTENT:")
#     print("-" * 40)
#     print(thinking_content if thinking_content else "[No thinking content found]")

#     print(f"\n FINAL CONTENT:")
#     print("-" * 40)
#     print(content)
    
#     # Performance metrics
#     input_tokens = len(model_inputs.input_ids[0])
#     output_tokens = len(generated_ids[0]) - input_tokens
#     total_tokens = len(generated_ids[0])
    
#     print(f"\n Performance Metrics:")
#     print(f"⏱️  Generation time: {generation_time:.2f}s")
#     print(f" Input tokens: {input_tokens}")
#     print(f" Output tokens: {output_tokens}")
#     print(f" Total tokens: {total_tokens}")
#     print(f" Tokens per second: {output_tokens / generation_time:.2f}")
    
#     print(f"\n Memory Summary:")
#     print(f"   CPU Memory - Before: {gen_start_cpu:.2f}GB, After: {gen_end_cpu:.2f}GB, Delta: {gen_end_cpu - gen_start_cpu:+.2f}GB")
#     print(f"   GPU Memory - Before: {gen_start_gpu:.2f}GB, After: {gen_end_gpu:.2f}GB, Delta: {gen_end_gpu - gen_start_gpu:+.2f}GB")
#     print(f"   GPU Reserved - Before: {gen_start_gpu_reserved:.2f}GB, After: {gen_end_gpu_reserved:.2f}GB, Delta: {gen_end_gpu_reserved - gen_start_gpu_reserved:+.2f}GB")
    
#     # KV Cache analysis (if available)
#     if hasattr(model, 'kv_cache_manager'):
#         print(f"\n Dynamic KV Cache Status:")
#         kv_manager = model.kv_cache_manager
#         print(f"   Current step: {kv_manager.current_step}")
#         print(f"   Window size: {kv_manager.window_size}")
#         print(f"   Decay factor: {kv_manager.decay_factor}")
#         print(f"   Enable dynamic KV: {getattr(kv_manager, 'enable_dynamic_kv', 'Unknown')}")
    
#     return {
#         'thinking_content': thinking_content,
#         'content': content,
#         'generation_time': generation_time,
#         'input_tokens': input_tokens,
#         'output_tokens': output_tokens,
#         'total_tokens': total_tokens,
#         'tokens_per_second': output_tokens / generation_time,
#         'cpu_memory_delta': gen_end_cpu - gen_start_cpu,
#         'gpu_memory_delta': gen_end_gpu - gen_start_gpu,
#         'gpu_reserved_delta': gen_end_gpu_reserved - gen_start_gpu_reserved,
#         'model_load_time': load_time,
#         'gpu_parameters_percent': gpu_params/total_params*100
#     }

# if __name__ == "__main__":
#     # Configuration - USE EXACT SAME AS ORIGINAL MODEL TEST
#     model_path = "C:\\Users\\ADMIN\\Desktop\\4B_shattered_shards_code\\architecture_implementation\\qwen3-4b-thinking-dynamic-kv"
#     prompt = """Give me a detailed explaination of machine learning"""
#     max_length = 512  # Same as original test
    
#     print(" Starting Dynamic KV Cache Model Inference")
#     print(f" Model path: {model_path}")
#     print(f" CUDA available: {torch.cuda.is_available()}")
#     if torch.cuda.is_available():
#         print(f" GPU: {torch.cuda.get_device_name(0)}")
#         print(f"易 GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
#         print(f" CUDA Version: {torch.version.cuda}")
    
#     # Clear GPU cache before starting
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         print("粒 GPU cache cleared before test")
    
#     # Run inference
#     results = dynamic_model_inference(model_path, prompt, max_length)
    
#     print("\n" + "=" * 60)
#     print("✅ Inference completed successfully!")
#     print("=" * 60)
    
#     # Final summary
#     print(f"\n FINAL RESULTS SUMMARY:")
#     print(f"⏱️  Model Loading: {results['model_load_time']:.2f}s")
#     print(f"⚡ Generation: {results['generation_time']:.2f}s")
#     print(f" Tokens/sec: {results['tokens_per_second']:.2f}")
#     print(f" Input tokens: {results['input_tokens']}")
#     print(f" Output tokens: {results['output_tokens']}")
#     print(f" Total tokens: {results['total_tokens']}")
#     print(f" CPU Memory Delta: {results['cpu_memory_delta']:+.2f}GB")
#     print(f" GPU Memory Delta: {results['gpu_memory_delta']:+.2f}GB")
#     print(f" GPU Reserved Delta: {results['gpu_reserved_delta']:+.2f}GB")
#     print(f" Parameters on GPU: {results['gpu_parameters_percent']:.1f}%")


# use_dynamic_model.py
