# download_and_save_dynamic_qwen.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import shutil

# Import your custom architecture - use absolute path
import sys
sys.path.append(r'C:\Users\ADMIN\Desktop\4B_shattered_shards_code\architecture_implementation')
from modeling_dynamic_qwen import monkey_patch_model_with_dynamic_kv, load_dynamic_model

def download_and_save_dynamic_model():
    """Download Qwen3-4B-thinking-2507 and save with dynamic KV cache"""
    
    # Configuration
    original_model_name = "Qwen/Qwen3-4B-thinking-2507"
    save_dir = "./Qwen3-4B-thinking-2507-dynamic-kv"
    
    # Path to your architecture file
    architecture_dir = r"C:\Users\ADMIN\Desktop\4B_shattered_shards_code\architecture_implementation"
    modeling_file_path = os.path.join(architecture_dir, "modeling_dynamic_qwen.py")
    
    # KV Cache configuration
    kv_config = {
        "decay_type": "exponential",
        "importance_metric": "learned",
        "window_size": 4096,
        "decay_factor": 0.97,
        "importance_power": 1.2,
        "min_decay": 0.15,
        "enable_importance": True,
        "enable_compression": False,
        "compression_ratio": 0.5,
        "enable_adaptive_window": True,
        "max_context_length": 8192,
        "min_window_size": 1024,
        "max_window_size": 8192,
        "adaptation_strategy": "hybrid"
    }
    
    print("🚀 Starting Dynamic Qwen3-4B-thinking-2507 Download & Save")
    print("=" * 60)
    print(f"📁 Architecture file: {modeling_file_path}")
    
    # Step 1: Download original model
    print("📥 Step 1: Downloading original model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            original_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            original_model_name,
            trust_remote_code=True
        )
        
        print("✅ Original model downloaded successfully!")
        print(f"📱 Model device: {model.device}")
        print(f"🧠 Model dtype: {model.dtype}")
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None
    
    # Step 2: Apply dynamic KV cache
    print("\n🔧 Step 2: Applying dynamic KV cache...")
    try:
        model = monkey_patch_model_with_dynamic_kv(model, kv_config)
        print("✅ Dynamic KV cache applied successfully!")
        
        # Test that KV cache manager is working
        if hasattr(model, 'kv_cache_manager'):
            stats = model.kv_cache_manager.get_cache_stats()
            print("📊 KV Cache Manager Status: Active")
            print(f"   - Window size: {stats['window_size']}")
            print(f"   - Adaptive window: {stats['enable_adaptive_window']}")
        else:
            print("❌ KV Cache Manager not found!")
            return None
            
    except Exception as e:
        print(f"❌ Error applying dynamic KV cache: {e}")
        return None
    
    # Step 3: Create save directory
    print(f"\n💾 Step 3: Saving model to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Save model weights
        model.save_pretrained(
            save_dir,
            safe_serialization=True,
            max_shard_size="2GB"
        )
        
        # Save tokenizer
        tokenizer.save_pretrained(save_dir)
        
        print("✅ Model weights and tokenizer saved!")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return None
    
    # Step 4: Update configuration
    print("\n⚙️ Step 4: Updating configuration...")
    try:
        config_path = os.path.join(save_dir, "config.json")
        
        # Read original config
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        
        # Add dynamic KV cache configuration
        config_data["dynamic_kv_cache"] = True
        config_data["kv_cache_manager_config"] = kv_config
        config_data["auto_map"] = {
            "AutoModelForCausalLM": "modeling_dynamic_qwen.load_dynamic_model"
        }
        
        # Save updated config
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print("✅ Configuration updated with dynamic KV cache settings!")
        
    except Exception as e:
        print(f"❌ Error updating configuration: {e}")
        return None
    
    # Step 5: Save architecture files
    print("\n📁 Step 5: Saving architecture files...")
    try:
        # Check if modeling file exists
        if not os.path.exists(modeling_file_path):
            print(f"❌ Architecture file not found at: {modeling_file_path}")
            print("📋 Please check the path and try again")
            return None
        
        # Copy your modeling_dynamic_qwen.py to the save directory
        target_script_path = os.path.join(save_dir, "modeling_dynamic_qwen.py")
        shutil.copy2(modeling_file_path, target_script_path)
        print("✅ modeling_dynamic_qwen.py copied!")
        
        # Create __init__.py for proper package structure
        init_content = '''"""
Dynamic KV Cache Qwen Model
"""

from .modeling_dynamic_qwen import (
    load_dynamic_model,
    monkey_patch_model_with_dynamic_kv,
    DynamicKVCacheManager,
    AdaptiveWindowManager
)

__all__ = [
    'load_dynamic_model',
    'monkey_patch_model_with_dynamic_kv', 
    'DynamicKVCacheManager',
    'AdaptiveWindowManager'
]
'''
        with open(os.path.join(save_dir, "__init__.py"), "w", encoding="utf-8") as f:
            f.write(init_content)
        print("✅ __init__.py created!")
        
    except Exception as e:
        print(f"❌ Error saving architecture files: {e}")
        return None
    
    # Step 6: Create usage documentation
    print("\n📖 Step 6: Creating usage documentation...")
    

download_and_save_dynamic_model()