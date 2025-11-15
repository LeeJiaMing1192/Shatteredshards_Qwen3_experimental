import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import psutil
import gc
import pandas as pd
import re
from typing import List, Dict, Tuple
import numpy as np
import csv
import os
from datetime import datetime

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

def clear_memory():
    """Clear GPU and CPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def parse_thinking_content_correct(output_ids, input_ids, tokenizer):
    """
    Correctly parse thinking content using the official method
    """
    # Extract only the generated tokens (exclude input)
    generated_ids = output_ids[len(input_ids[0]):].tolist()
    
    # Look for the </think> token (151668) in the generated output
    try:
        # Find the last occurrence of </think> token using rindex method
        index = len(generated_ids) - generated_ids[::-1].index(151668)
        
        # Everything before </think> is thinking content (excluding <think> if present)
        thinking_content_ids = generated_ids[:index]
        
        # Remove <think> token (151667) from the beginning if present
        if thinking_content_ids and thinking_content_ids[0] == 151667:
            thinking_content_ids = thinking_content_ids[1:]
            
        thinking_content = tokenizer.decode(thinking_content_ids, skip_special_tokens=True).strip("\n")
        
        # Everything after </think> is the final content
        final_content_ids = generated_ids[index:]
        
        # Remove </think> token from the beginning if present
        if final_content_ids and final_content_ids[0] == 151668:
            final_content_ids = final_content_ids[1:]
            
        final_content = tokenizer.decode(final_content_ids, skip_special_tokens=True).strip("\n")
        
    except ValueError:
        # If </think> token not found, consider everything as thinking content
        thinking_content_ids = generated_ids
        
        # Remove <think> token from the beginning if present
        if thinking_content_ids and thinking_content_ids[0] == 151667:
            thinking_content_ids = thinking_content_ids[1:]
            
        thinking_content = tokenizer.decode(thinking_content_ids, skip_special_tokens=True).strip("\n")
        final_content = ""
    
    return thinking_content, final_content

def extract_final_answer_from_content(content: str) -> str:
    """
    Extract the final answer from the content after </think>
    For multiple choice questions, we look for the selected option (A, B, C, D, etc.)
    """
    if not content:
        return None
    
    # Look for multiple choice patterns in the final content
    patterns = [
        r'ANSWER:\s*([A-D])',  # Matches "ANSWER: A"
        r'Answer:\s*([A-D])',  # Matches "Answer: A"
        r'answer:\s*([A-D])',  # Matches "answer: A"
        r'Final answer:\s*([A-D])',  # Matches "Final answer: A"
        r'final answer:\s*([A-D])',  # Matches "final answer: A"
        r'The answer is\s*([A-D])',  # Matches "The answer is A"
        r'the answer is\s*([A-D])',  # Matches "the answer is A"
        r'####\s*([A-D])',  # Matches "#### A"
        r'Selected option:\s*([A-D])',  # Matches "Selected option: A"
        r'Option\s*([A-D])\s*is correct',  # Matches "Option A is correct"
        r'Choice\s*([A-D])\s*is correct',  # Matches "Choice A is correct"
        r'I choose\s*([A-D])',  # Matches "I choose A"
        r'I select\s*([A-D])',  # Matches "I select A"
    ]
    
    content_clean = content.strip()
    
    # Check patterns in order
    for pattern in patterns:
        matches = re.findall(pattern, content_clean, re.IGNORECASE)
        if matches:
            return matches[-1].upper()  # Return the last match and ensure uppercase
    
    # If no pattern found, look for single letter A-D in the content
    single_letter_match = re.search(r'\b([A-D])\b', content_clean)
    if single_letter_match:
        return single_letter_match.group(1).upper()
    
    return None

def format_question_with_choices(question: str, correct_answer: str, incorrect_1: str, incorrect_2: str, incorrect_3: str) -> str:
    """
    Format the question with shuffled answer choices for the model
    The correct answer is mixed with incorrect answers in random order
    """
    # Create a list of all answers and shuffle them
    import random
    answers = [
        (correct_answer, 'A'),
        (incorrect_1, 'B'), 
        (incorrect_2, 'C'),
        (incorrect_3, 'D')
    ]
    
    # Shuffle the answers but keep track of the correct one
    random.shuffle(answers)
    
    # Build the answer choices string
    answer_choices = "Please choose the correct answer from the following options:\n"
    for answer_text, letter in answers:
        answer_choices += f"{letter}. {answer_text}\n"
    
    prompt = f"""{question}

{answer_choices}
Please think through this problem step by step and provide your reasoning. Then, give your final answer in the format: "ANSWER: X" where X is the letter of the correct choice (A, B, C, or D)."""
    
    return prompt, answers  # Return the shuffled answers to track correct letter

def calculate_pass_at_1(results: List[Dict]) -> float:
    """
    Calculate pass@1 accuracy
    """
    correct = sum(1 for result in results if result['correct'])
    total = len(results)
    return correct / total if total > 0 else 0.0

def create_csv_logger(filename: str):
    """Create CSV logger with comprehensive headers"""
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    headers = [
        'timestamp', 'question_id', 'question_length', 'ground_truth', 'predicted_answer', 
        'correct', 'pass_at_1', 'input_tokens', 'output_tokens', 'total_tokens',
        'time_to_first_token', 'generation_time', 'total_time', 'encode_time', 
        'decode_time', 'tokens_per_second', 'ram_usage_mb', 'gpu_allocated_mb', 
        'gpu_reserved_mb', 'has_thinking', 'thinking_tokens', 'final_tokens',
        'thinking_length', 'final_length', 'model_name', 'batch_size', 'temperature',
        'thinking_content_preview', 'final_content_preview', 'subject_domain',
        'correct_answer_text', 'shuffled_correct_letter'
    ]
    
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
    
    return headers

def log_to_csv(filename: str, data: Dict, headers: List[str]):
    """Log data to CSV file"""
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow(data)

def measure_time_to_first_token(model, model_inputs, tokenizer):
    """Measure time to first token using a small generation"""
    start_time = time.time()
    with torch.no_grad():
        first_tokens = model.generate(
            **model_inputs,
            max_new_tokens=1,  # Just get first token
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    first_token_time = time.time() - start_time
    return first_token_time

def validate_csv_structure(file_path: str) -> Tuple[bool, List[str], pd.DataFrame]:
    """
    Validate the CSV file structure and return sample data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Successfully loaded CSV file with {len(df)} rows")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Check for expected columns
        expected_columns = ['Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False, list(df.columns), df.head(3)
        
        print("✅ All expected columns present")
        print("\n📋 Sample data:")
        print(df.head(3))
        
        # Check data format
        question_sample = df['Question'].iloc[0]
        correct_sample = df['Correct Answer'].iloc[0]
        incorrect_1_sample = df['Incorrect Answer 1'].iloc[0]
        
        print(f"📝 Question sample: '{question_sample[:100]}...'")
        print(f"✅ Correct answer sample: '{correct_sample}'")
        print(f"❌ Incorrect answer 1 sample: '{incorrect_1_sample}'")
        
        return True, list(df.columns), df.head(3)
        
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return False, [], pd.DataFrame()

def benchmark_model():
    """Benchmark the model on the GPQA diamond dataset with comprehensive CSV logging"""
    
    model_path = "C:\\Users\\ADMIN\\Desktop\\4B_shattered_shards_code\\Qwen3-4B-thinking-2507-dynamic-kv"
    csv_path = "C:\\Users\\ADMIN\\Downloads\\gpqa_diamond.csv"
    
    # Create results directory
    results_dir = "benchmark_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create CSV filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(results_dir, f"gpqa_benchmark_results_{timestamp}.csv")
    
    print("🧪 MODEL BENCHMARKING WITH GPQA DIAMOND DATASET")
    print("=" * 60)
    print(f"📁 Results will be saved to: {csv_filename}")
    
    # Validate CSV file
    print("📁 Validating CSV file...")
    is_valid, columns, sample_data = validate_csv_structure(csv_path)
    
    if not is_valid:
        print("❌ Invalid CSV file structure. Please check the file.")
        return
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} questions for benchmarking")
    
    # Initial memory
    initial_ram, initial_gpu = get_memory_usage()
    
    # Load tokenizer and model
    print("\n🔧 Loading model and tokenizer...")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    load_time = time.time() - start_time
    
    print(f"⏱️ Model load time: {load_time:.2f}s")
    
    # Create CSV logger
    csv_headers = create_csv_logger(csv_filename)
    
    # Benchmarking parameters
    max_samples = min(50, len(df))  # Limit to first 50 samples for quick testing
    print(f"\n🎯 Benchmarking on {max_samples} samples...")
    
    results = []
    generation_stats = []
    cumulative_correct = 0
    
    for idx, (_, row) in enumerate(df.head(max_samples).iterrows()):
        if idx >= max_samples:
            break
            
        print(f"\n{'='*50}")
        print(f"Question {idx + 1}/{max_samples}")
        print(f"{'='*50}")
        
        question = row['Question']
        correct_answer = row['Correct Answer']
        incorrect_1 = row['Incorrect Answer 1']
        incorrect_2 = row['Incorrect Answer 2']
        incorrect_3 = row['Incorrect Answer 3']
        subject_domain = row.get('Subject/Domain', 'Unknown')
        
        # Format the prompt with shuffled answer choices
        prompt, shuffled_answers = format_question_with_choices(
            question, correct_answer, incorrect_1, incorrect_2, incorrect_3
        )
        
        # Find which letter corresponds to the correct answer after shuffling
        correct_letter = None
        for answer_text, letter in shuffled_answers:
            if answer_text == correct_answer:
                correct_letter = letter
                break
        
        print(f"📝 Question: {question[:150]}...")
        print(f"🔠 Shuffled options:")
        for answer_text, letter in shuffled_answers:
            marker = " ✅" if letter == correct_letter else ""
            print(f"   {letter}. {answer_text}{marker}")
        print(f"✅ Correct letter: {correct_letter}")
        print(f"📚 Subject: {subject_domain}")
        
        # Clear cache before each question
        clear_memory()
        ram_before, gpu_before = get_memory_usage()
        
        # Prepare chat template
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template and tokenize
        encode_start = time.time()
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        encode_time = time.time() - encode_start
        
        # Measure time to first token
        first_token_time = measure_time_to_first_token(model, model_inputs, tokenizer)
        
        # Generate full response
        gen_start = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=5000,
                do_sample=False,  # Use greedy decoding for consistent benchmarking
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_time = time.time() - gen_start
        
        # Parse thinking content using the official method
        decode_start = time.time()
        thinking_content, final_content = parse_thinking_content_correct(generated_ids[0], model_inputs.input_ids, tokenizer)
        decode_time = time.time() - decode_start
        
        total_time = encode_time + gen_time + decode_time
        
        # Extract final answer from the final content (after </think>)
        predicted_answer = extract_final_answer_from_content(final_content)
        
        # Check if correct (letter comparison)
        is_correct = False
        if predicted_answer and correct_letter:
            is_correct = (predicted_answer.strip().upper() == correct_letter.strip().upper())
        
        cumulative_correct += 1 if is_correct else 0
        current_pass_at_1 = cumulative_correct / (idx + 1)
        
        # Memory after generation
        ram_after, gpu_after = get_memory_usage()
        
        # Calculate tokens
        input_tokens = model_inputs.input_ids.shape[1]
        output_tokens = generated_ids.shape[1] - input_tokens
        total_tokens = input_tokens + output_tokens
        tokens_per_sec = output_tokens / gen_time if gen_time > 0 else 0
        
        # Calculate thinking vs final tokens
        thinking_tokens = len(tokenizer.encode(thinking_content)) if thinking_content else 0
        final_tokens = len(tokenizer.encode(final_content)) if final_content else 0
        
        # Prepare CSV log data
        csv_data = {
            'timestamp': datetime.now().isoformat(),
            'question_id': idx,
            'question_length': len(question),
            'ground_truth': correct_letter,
            'predicted_answer': predicted_answer or 'NONE',
            'correct': is_correct,
            'pass_at_1': current_pass_at_1,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'time_to_first_token': first_token_time,
            'generation_time': gen_time,
            'total_time': total_time,
            'encode_time': encode_time,
            'decode_time': decode_time,
            'tokens_per_second': tokens_per_sec,
            'ram_usage_mb': ram_after,
            'gpu_allocated_mb': gpu_after.get('GPU_0', {}).get('allocated', 0) if gpu_after else 0,
            'gpu_reserved_mb': gpu_after.get('GPU_0', {}).get('reserved', 0) if gpu_after else 0,
            'has_thinking': bool(thinking_content),
            'thinking_tokens': thinking_tokens,
            'final_tokens': final_tokens,
            'thinking_length': len(thinking_content),
            'final_length': len(final_content),
            'model_name': os.path.basename(model_path),
            'batch_size': 1,
            'temperature': 0.7,
            'thinking_content_preview': thinking_content[:100] if thinking_content else '',
            'final_content_preview': final_content[:100] if final_content else '',
            'subject_domain': subject_domain,
            'correct_answer_text': correct_answer,
            'shuffled_correct_letter': correct_letter
        }
        
        # Log to CSV
        log_to_csv(csv_filename, csv_data, csv_headers)
        
        # Store results for summary
        result = {
            'question_id': idx,
            'ground_truth': correct_letter,
            'predicted_answer': predicted_answer,
            'correct': is_correct,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'gen_time': gen_time,
            'total_time': total_time,
            'tokens_per_sec': tokens_per_sec,
            'first_token_time': first_token_time,
            'thinking_content': thinking_content,
            'final_content': final_content,
            'subject_domain': subject_domain,
            'correct_answer_text': correct_answer,
            'shuffled_correct_letter': correct_letter
        }
        results.append(result)
        
        # Print results for this question
        print(f"🤔 Thinking: {thinking_content[:100]}..." if thinking_content else "🤔 Thinking: [None]")
        print(f"💬 Final Content: {final_content[:100]}..." if final_content else "💬 Final Content: [None]")
        print(f"🎯 Predicted: {predicted_answer} | Ground Truth: {correct_letter} | ✅ Correct: {is_correct}")
        print(f"⏱️ Time to first token: {first_token_time:.3f}s | Generation: {gen_time:.2f}s | Total: {total_time:.2f}s")
        print(f"⚡ Tokens/s: {tokens_per_sec:.2f} | Pass@1: {current_pass_at_1:.3f}")
        print(f"💾 RAM: {ram_after:.0f}MB | GPU: {gpu_after.get('GPU_0', {}).get('allocated', 0):.0f}MB")
        
        generation_stats.append({
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'gen_time': gen_time,
            'first_token_time': first_token_time,
            'tokens_per_sec': tokens_per_sec,
            'ram_usage': ram_after
        })
    
    # Calculate overall metrics
    print("\n" + "="*60)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    if results:
        pass_at_1 = calculate_pass_at_1(results)
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r['correct'])
        unanswered = sum(1 for r in results if r['predicted_answer'] is None)
        
        # Performance statistics
        if generation_stats:
            avg_tokens_per_sec = np.mean([stat['tokens_per_sec'] for stat in generation_stats])
            avg_gen_time = np.mean([stat['gen_time'] for stat in generation_stats])
            avg_first_token_time = np.mean([stat['first_token_time'] for stat in generation_stats])
            avg_input_tokens = np.mean([stat['input_tokens'] for stat in generation_stats])
            avg_output_tokens = np.mean([stat['output_tokens'] for stat in generation_stats])
            max_ram_usage = max([stat['ram_usage'] for stat in generation_stats])
            
            print(f"🎯 Pass@1 Accuracy: {pass_at_1:.4f} ({correct_answers}/{total_questions})")
            print(f"❓ Unanswered: {unanswered}/{total_questions}")
            
            print(f"\n⚡ Performance Statistics:")
            print(f"📊 Average tokens per second: {avg_tokens_per_sec:.2f}")
            print(f"⏱️ Average generation time: {avg_gen_time:.2f}s")
            print(f"🚀 Average time to first token: {avg_first_token_time:.3f}s")
            print(f"🔢 Average input tokens: {avg_input_tokens:.1f}")
            print(f"🔢 Average output tokens: {avg_output_tokens:.1f}")
            print(f"💾 Peak RAM usage: {max_ram_usage:.0f} MB")
            
            # Calculate accuracy by subject domain
            subject_results = {}
            for result in results:
                subject = result['subject_domain']
                if subject not in subject_results:
                    subject_results[subject] = {'total': 0, 'correct': 0}
                subject_results[subject]['total'] += 1
                if result['correct']:
                    subject_results[subject]['correct'] += 1
            
            print(f"\n📚 Accuracy by Subject Domain:")
            for subject, stats in subject_results.items():
                accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                print(f"  {subject}: {accuracy:.3f} ({stats['correct']}/{stats['total']})")
    
    # Final memory usage
    final_ram, final_gpu = get_memory_usage()
    print(f"\n💾 Final RAM usage: {final_ram:.0f} MB")
    if final_gpu:
        for device, memory in final_gpu.items():
            print(f"💾 {device}: Allocated: {memory['allocated']:.0f} MB, Reserved: {memory['reserved']:.0f} MB")
    
    print(f"\n💾 Results saved to: {csv_filename}")
    
    # Save summary statistics
    if results:
        summary_filename = os.path.join(results_dir, f"gpqa_benchmark_summary_{timestamp}.txt")
        with open(summary_filename, 'w') as f:
            f.write("GPQA DIAMOND BENCHMARK SUMMARY\n")
            f.write("==============================\n\n")
            f.write(f"Model: {os.path.basename(model_path)}\n")
            f.write(f"Dataset: {os.path.basename(csv_path)}\n")
            f.write(f"Total questions: {total_questions}\n")
            f.write(f"Pass@1 Accuracy: {pass_at_1:.4f} ({correct_answers}/{total_questions})\n")
            f.write(f"Unanswered: {unanswered}\n")
            f.write(f"Average tokens/sec: {avg_tokens_per_sec:.2f}\n")
            f.write(f"Average generation time: {avg_gen_time:.2f}s\n")
            f.write(f"Average time to first token: {avg_first_token_time:.3f}s\n")
            f.write(f"Peak RAM usage: {max_ram_usage:.0f} MB\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Write subject domain breakdown
            f.write(f"\nAccuracy by Subject Domain:\n")
            for subject, stats in subject_results.items():
                accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                f.write(f"  {subject}: {accuracy:.3f} ({stats['correct']}/{stats['total']})\n")
        
        print(f"📄 Summary saved to: {summary_filename}")
    
    # Clean up
    print("\n🧹 Cleaning up...")
    del model
    del tokenizer
    clear_memory()

if __name__ == "__main__":
    benchmark_model()