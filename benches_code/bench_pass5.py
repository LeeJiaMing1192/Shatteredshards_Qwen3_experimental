import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import csv
import os
from tqdm import tqdm
import psutil
import pyarrow.parquet as pq

def get_gpu_memory():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        return allocated, reserved
    return 0, 0

def get_system_memory():
    """Get system memory usage"""
    process = psutil.Process()
    memory_gb = process.memory_info().rss / 1024**3  # GB
    return memory_gb

def calculate_accuracy(prediction, reference):
    """Calculate basic accuracy (Pass@1)"""
    prediction_clean = prediction.strip().lower()
    reference_clean = reference.strip().lower()
    return int(prediction_clean == reference_clean)

def calculate_pass_at_5(predictions, reference):
    """Calculate Pass@5 - check if reference matches any of the 5 predictions"""
    reference_clean = reference.strip().lower()
    for prediction in predictions:
        prediction_clean = prediction.strip().lower()
        if prediction_clean == reference_clean:
            return 1
    return 0

def calculate_mr_score(prediction, reference):
    """Calculate MR-Score (Match Rate Score)"""
    pred_words = set(prediction.strip().lower().split())
    ref_words = set(reference.strip().lower().split())
    
    if not ref_words:
        return 0.0
    
    intersection = pred_words.intersection(ref_words)
    return len(intersection) / len(ref_words)

def calculate_mr_score_pass5(predictions, reference):
    """Calculate MR-Score for Pass@5 - take the best MR score from 5 predictions"""
    reference_clean = reference.strip().lower()
    ref_words = set(reference_clean.split())
    
    if not ref_words:
        return 0.0
    
    best_mr_score = 0.0
    for prediction in predictions:
        prediction_clean = prediction.strip().lower()
        pred_words = set(prediction_clean.split())
        
        intersection = pred_words.intersection(ref_words)
        mr_score = len(intersection) / len(ref_words)
        best_mr_score = max(best_mr_score, mr_score)
    
    return best_mr_score

def load_dynamic_model(model_path):
    """
    Load your dynamically modified Qwen model from local path
    """
    print(f"⚡ Loading dynamic model from local path: {model_path}")
    
    # For local models, you need to specify local_files_only=True
    # and ensure the path exists
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True  # This tells it to look for local files only
    )
    return model

def load_parquet_dataset(parquet_path):
    """Load dataset from parquet file"""
    try:
        print(f"📂 Loading parquet file from: {parquet_path}")
        
        # Read parquet file
        table = pq.read_table(parquet_path)
        dataset = table.to_pandas()
        
        print(f"✅ Successfully loaded dataset with {len(dataset)} rows")
        print(f"📊 Columns available: {list(dataset.columns)}")
        
        # Check if required columns exist
        if 'question' not in dataset.columns or 'answer' not in dataset.columns:
            print("❌ Required columns 'question' and 'answer' not found.")
            print("   Available columns:", list(dataset.columns))
            
            # Try to find alternative column names
            question_cols = [col for col in dataset.columns if 'question' in col.lower() or 'prompt' in col.lower()]
            answer_cols = [col for col in dataset.columns if 'answer' in col.lower() or 'response' in col.lower() or 'output' in col.lower()]
            
            if question_cols and answer_cols:
                print(f"🔍 Using alternative columns: {question_cols[0]} as question, {answer_cols[0]} as answer")
                dataset = dataset.rename(columns={
                    question_cols[0]: 'question',
                    answer_cols[0]: 'answer'
                })
            else:
                print("❓ Using first two columns as question and answer")
                if len(dataset.columns) >= 2:
                    dataset = dataset.rename(columns={
                        dataset.columns[0]: 'question',
                        dataset.columns[1]: 'answer'
                    })
                else:
                    raise ValueError("Not enough columns in the dataset")
        
        print(f"📝 Sample question: {dataset['question'].iloc[0][:100]}...")
        print(f"📝 Sample answer: {dataset['answer'].iloc[0][:100]}...")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error loading parquet file: {e}")
        print("📋 Creating fallback dataset...")
        
        # Create fallback dataset
        dataset = pd.DataFrame({
            'question': [
                'What is the capital of France?',
                'Explain quantum computing in simple terms',
                'What is machine learning?',
                'How does a neural network work?',
                'What is artificial intelligence?'
            ],
            'answer': [
                'Paris',
                'Quantum computing uses quantum bits that can exist in multiple states simultaneously',
                'Machine learning is a subset of AI that enables computers to learn without explicit programming',
                'Neural networks are computing systems inspired by biological neural networks in brains',
                'Artificial intelligence is the simulation of human intelligence processes by machines'
            ]
        })
        return dataset

def ensure_results_folder():
    """Create results folder if it doesn't exist"""
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print(f"📁 Created results folder: {results_folder}")
    return results_folder

def evaluate_models_on_dataset():
    """Evaluate both models on GMSK-8 parquet dataset"""
    
    # Create results folder
    results_folder = ensure_results_folder()
    
    # Load dataset from parquet
    parquet_path = "C:\\Users\\ADMIN\\Downloads\\0000 (1).parquet"
    dataset = load_parquet_dataset(parquet_path)
    
    print(f"📊 Dataset loaded with {len(dataset)} samples")
    
    # Limit dataset size if too large to avoid memory issues
    max_samples = 100  # Adjust based on your needs
    if len(dataset) > max_samples:
        print(f"📦 Limiting to {max_samples} samples for evaluation")
        dataset = dataset.head(max_samples)
    
    # Initialize models
    print("🔄 Loading models...")
    
    # Normal model
    model_normal = AutoModelForCausalLM.from_pretrained(
        "Google/gemma-2-2b-it",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Google/gemma-2-2b-it")
    
    # Dynamic model - using the same path for demonstration
    # Replace with your actual dynamic model path if different
    dynamic_model_path = ""
    model_dynamic = load_dynamic_model(dynamic_model_path)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # CSV file setup - now in results folder
    normal_csv = os.path.join(results_folder, "Gemma_2_2B_ITpass5.csv")
    dynamic_csv = os.path.join(results_folder, "HYDEIKV-Gemma_2_2B_IT_pass5.csv")
    
    # CSV headers
    headers = [
        'question', 'reference_answer', 'generated_answer_1', 'generated_answer_2', 
        'generated_answer_3', 'generated_answer_4', 'generated_answer_5',
        'prompt_length', 'output_length', 'first_token_time_ms',
        'total_generation_time_ms', 'time_per_output_token_ms',
        'memory_allocated_gb', 'memory_reserved_gb', 'system_memory_gb',
        'accuracy_pass5', 'mr_score_pass5', 'tokens_per_second'
    ]
    
    # Initialize CSV files
    for csv_file in [normal_csv, dynamic_csv]:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def evaluate_single_model(model, model_name, csv_file):
        """Evaluate a single model on the dataset with Pass@5"""
        print(f"\n🔍 Evaluating {model_name}...")
        
        results = []
        
        for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Processing {model_name}"):
            try:
                question = str(row['question'])
                reference_answer = str(row['answer'])
                
                # Tokenize input
                inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                prompt_length = inputs['input_ids'].shape[1]
                
                # Clear cache and synchronize
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Measure first token time using a short generation
                start_time = time.time()
                with torch.no_grad():
                    # Generate just 1 token to measure first token time
                    first_token_output = model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                first_token_time = time.time() - start_time
                
                # Generate 5 different responses for Pass@5
                all_generated_answers = []
                total_generation_time = 0
                total_output_length = 0
                
                for i in range(5):
                    gen_start_time = time.time()
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id,
                            num_return_sequences=1
                        )
                    generation_time = time.time() - gen_start_time
                    total_generation_time += generation_time
                    
                    # Decode output
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated_answer = generated_text[len(question):].strip()
                    all_generated_answers.append(generated_answer)
                    
                    output_length = len(outputs[0]) - prompt_length
                    total_output_length += output_length
                
                # Calculate average metrics across 5 generations
                avg_output_length = total_output_length / 5
                avg_time_per_token = (total_generation_time * 1000) / total_output_length if total_output_length > 0 else 0
                avg_tokens_per_second = total_output_length / total_generation_time if total_generation_time > 0 else 0
                
                # Calculate Pass@5 metrics
                pass5_accuracy = calculate_pass_at_5(all_generated_answers, reference_answer)
                pass5_mr_score = calculate_mr_score_pass5(all_generated_answers, reference_answer)
                
                # Get memory usage
                mem_alloc, mem_reserved = get_gpu_memory()
                sys_memory = get_system_memory()
                
                # Store results
                result = {
                    'question': question,
                    'reference_answer': reference_answer,
                    'generated_answer_1': all_generated_answers[0],
                    'generated_answer_2': all_generated_answers[1],
                    'generated_answer_3': all_generated_answers[2],
                    'generated_answer_4': all_generated_answers[3],
                    'generated_answer_5': all_generated_answers[4],
                    'prompt_length': prompt_length,
                    'output_length': avg_output_length,
                    'first_token_time_ms': first_token_time * 1000,
                    'total_generation_time_ms': total_generation_time * 1000,
                    'time_per_output_token_ms': avg_time_per_token,
                    'memory_allocated_gb': mem_alloc,
                    'memory_reserved_gb': mem_reserved,
                    'system_memory_gb': sys_memory,
                    'accuracy_pass5': pass5_accuracy,
                    'mr_score_pass5': pass5_mr_score,
                    'tokens_per_second': avg_tokens_per_second
                }
                
                results.append(result)
                
                # Write to CSV immediately
                with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([result[header] for header in headers])
                
                # Clear cache between samples
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Error processing sample {idx} with {model_name}: {e}")
                continue
        
        return results
    
    # Evaluate both models
    print("🚀 Starting evaluation...")
    
    # Evaluate normal model
    normal_results = evaluate_single_model(model_normal, "Normal Gemma 2 2B it", normal_csv)
    
    # Clear memory before loading dynamic model
    del model_normal
    torch.cuda.empty_cache()
    time.sleep(2)  # Give time for memory cleanup
    
    # Evaluate dynamic model
    dynamic_results = evaluate_single_model(model_dynamic, "Dynamic Gemma 2 2B it", dynamic_csv)
    
    # Clean up
    del model_dynamic
    torch.cuda.empty_cache()
    
    # Calculate and display summary statistics
    print("\n" + "=" * 80)
    print("📊 SUMMARY STATISTICS (Pass@5)")
    print("=" * 80)
    
    def calculate_summary_stats(results, model_name):
        if not results:
            print(f"\n❌ No results for {model_name}")
            return
        
        df = pd.DataFrame(results)
        print(f"\n📈 {model_name} Performance Summary:")
        print(f"   • Samples Processed: {len(df)}")
        print(f"   • Average First Token Time: {df['first_token_time_ms'].mean():.2f} ms")
        print(f"   • Average Total Generation Time: {df['total_generation_time_ms'].mean():.2f} ms")
        print(f"   • Average TPOT: {df['time_per_output_token_ms'].mean():.2f} ms/token")
        print(f"   • Average Tokens/Second: {df['tokens_per_second'].mean():.2f}")
        print(f"   • Average Accuracy (Pass@5): {df['accuracy_pass5'].mean():.3f}")
        print(f"   • Average MR-Score (Pass@5): {df['mr_score_pass5'].mean():.3f}")
        print(f"   • Average GPU Memory: {df['memory_allocated_gb'].mean():.2f} GB")
        
        # Additional statistics
        print(f"   • Median TPOT: {df['time_per_output_token_ms'].median():.2f} ms/token")
        print(f"   • Max GPU Memory: {df['memory_allocated_gb'].max():.2f} GB")
    
    calculate_summary_stats(normal_results, "Normal Qwen 1.7B")
    calculate_summary_stats(dynamic_results, "Dynamic KV Qwen 1.7B")
    
    print(f"\n💾 Results saved to:")
    print(f"   • Normal model: {normal_csv}")
    print(f"   • Dynamic model: {dynamic_csv}")

def quick_comparison():
    """Quick side-by-side comparison with Pass@5"""
    prompt = "Explain the concept of artificial intelligence in simple terms."
    
    print("🔍 QUICK COMPARISON: NORMAL vs DYNAMIC KV MODEL (Pass@5)")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print("=" * 60)
    
    # Create results folder
    results_folder = ensure_results_folder()
    
    # Test Normal Model
    print("\n🧪 TESTING NORMAL MODEL...")
    start_time = time.time()
    
    model_normal = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model_normal.device)
    
    normal_responses = []
    for i in range(5):
        with torch.no_grad():
            outputs_normal = model_normal.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        normal_text = tokenizer.decode(outputs_normal[0], skip_special_tokens=True)
        normal_responses.append(normal_text[len(prompt):].strip())
    
    normal_time = time.time() - start_time
    
    # Clean up
    del model_normal, inputs, outputs_normal
    torch.cuda.empty_cache()
    
    # Test Dynamic Model
    print("\n⚡ TESTING DYNAMIC KV MODEL...")
    start_time = time.time()
    
    dynamic_model_path = "C:\\Users\\ADMIN\\Desktop\\memory_decay_testing_lib\\qwen-dynamic-kv-1.7B"
    model_dynamic = load_dynamic_model(dynamic_model_path)
    inputs = tokenizer(prompt, return_tensors="pt").to(model_dynamic.device)
    
    dynamic_responses = []
    for i in range(5):
        with torch.no_grad():
            outputs_dynamic = model_dynamic.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        dynamic_text = tokenizer.decode(outputs_dynamic[0], skip_special_tokens=True)
        dynamic_responses.append(dynamic_text[len(prompt):].strip())
    
    dynamic_time = time.time() - start_time
    
    # Clean up
    del model_dynamic, inputs, outputs_dynamic
    torch.cuda.empty_cache()
    
    # Save quick comparison results
    quick_csv = os.path.join(results_folder, "quick_comparison_pass5.csv")
    with open(quick_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'prompt', 'response_1', 'response_2', 'response_3', 'response_4', 'response_5', 'time_taken'])
        writer.writerow(['normal', prompt] + normal_responses + [normal_time])
        writer.writerow(['dynamic', prompt] + dynamic_responses + [dynamic_time])
    
    # Display Results
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    
    print(f"\n⏰ Normal Model Time: {normal_time:.2f}s")
    print(f"⚡ Dynamic Model Time: {dynamic_time:.2f}s")
    print(f"🚀 Speed Improvement: {((normal_time - dynamic_time) / normal_time * 100):+.1f}%")
    
    if torch.cuda.is_available():
        print(f"\n💾 GPU Memory Info:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
    
    print(f"\n📝 NORMAL MODEL OUTPUTS (Pass@5):")
    print("-" * 40)
    for i, response in enumerate(normal_responses, 1):
        print(f"Response {i}: {response[:100]}...")
    
    print(f"\n📝 DYNAMIC KV MODEL OUTPUTS (Pass@5):")
    print("-" * 40)
    for i, response in enumerate(dynamic_responses, 1):
        print(f"Response {i}: {response[:100]}...")
    
    print(f"\n💾 Quick comparison saved to: {quick_csv}")

if __name__ == "__main__":
    # Run comprehensive evaluation
    evaluate_models_on_dataset()
    
    # Run quick comparison
    print("\n" + "=" * 80)
    print("🚀 RUNNING QUICK COMPARISON (Pass@5)")
    print("=" * 80)
    quick_comparison()