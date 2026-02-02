"""Verification script for MinerU2.5 model download."""
import os
import sys
from transformers import AutoTokenizer

def verify_model_integrity(model_path: str):
    """
    Verifies that the model artifacts at the given path can be loaded.
    
    Args:
        model_path (str): Path to the local model directory
    """
    print(f"Checking model at: {model_path}")
    
    # 1. Check file existence
    required_files = ["model.safetensors", "config.json", "tokenizer.json"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    
    if missing:
        print(f"❌ Verification FAILED. Missing files: {missing}")
        sys.exit(1)
    
    print("✅ Vital files present.")
    
    # 2. logical check - Load Tokenizer
    try:
        print("Attempting to load Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        print(f"✅ Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"❌ Tvoerification FAILED. Could not load tokenizer: {e}")
        sys.exit(1)

    print("✅ Model download verified successfully.")

if __name__ == "__main__":
    MODEL_PATH = "./models/MinerU2.5"
    verify_model_integrity(MODEL_PATH)
