"""Verification script v2 for MinerU2.5 - Checks Model Loading."""
import os
import sys

def verify_model_integrity_v2(model_path: str):
    print(f"Checking model at: {model_path}")
    
    # 1. Dependency Check
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
        print(f"✅ PyTorch version: {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
    except ImportError as e:
        print(f"❌ Dependency Error: {e}")
        print("Please run: pip install torch torchvision transformers")
        return
    except Exception as e:
        print(f"❌ Unexpected Import Error: {e}")
        return

    # 2. Load Tokenizer
    try:
        print("Attempting to load Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        print(f"✅ Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ Tokenizer load failed: {e}")
        sys.exit(1)

    # 3. Load Model (CPU)
    try:
        print("Attempting to load Model Weights (CPU)... This may take 1-2 minutes...")
        # Qwen2VL is a vision language model, usually loaded with AutoModel or specific class
        # We use AutoModel to let transformers decide.
        model = AutoModel.from_pretrained(
            model_path, 
            local_files_only=True, 
            trust_remote_code=True,
            device_map="cpu", 
            low_cpu_mem_usage=True
        )
        print(f"✅ Model loaded successfully on CPU!")
        print(f"   Architecture: {model.config.architectures}")
        print(f"   Parameters: {model.num_parameters():,}")
    except Exception as e:
         print(f"❌ Model load failed: {e}")
         print("Note: If 'flash_attn' error, it's expected on CPU without custom kernels.")
         sys.exit(1)

    print("✅ FULL SYSTEM READY.")

if __name__ == "__main__":
    MODEL_PATH = "./models/MinerU2.5"
    verify_model_integrity_v2(MODEL_PATH)
