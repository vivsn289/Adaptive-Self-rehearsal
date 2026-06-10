#!/bin/bash
set -e

echo "=========================================="
echo "Self-Instruct Forgetting Experiment Setup"
echo "=========================================="

# Create virtual environment (using virtualenv since python3-venv may not be available)
if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    if [ ! -f "$HOME/.local/bin/virtualenv" ]; then
        echo "Installing virtualenv..."
        pip install --user virtualenv
    fi
    ~/.local/bin/virtualenv env
fi
source env/bin/activate

# Install dependencies
echo "Installing dependencies (this may take several minutes)..."
pip install --upgrade pip --quiet
pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
pip install transformers accelerate datasets trl peft lm-eval rouge-score matplotlib pandas tabulate bitsandbytes sentencepiece protobuf numpy --quiet

# HuggingFace authentication
echo ""
echo "=========================================="
echo "HuggingFace Authentication Required"
echo "=========================================="
echo "Before downloading Mistral-7B, you must:"
echo "  1. Accept the Mistral-7B license at:"
echo "     https://huggingface.co/mistralai/Mistral-7B-v0.1"
echo "  2. Then log in by running:"
echo "     huggingface-cli login"
echo "     (paste your HuggingFace token when prompted)"
echo "=========================================="
echo ""

# Clone Self-Instruct repo for seed tasks
if [ ! -d "self-instruct" ]; then
    echo "Cloning Self-Instruct repo for seed tasks..."
    git clone https://github.com/yizhongw/self-instruct.git
fi

# Create directory structure
mkdir -p results/base results/instruct results/figures generated_data logs

# Verify GPU
echo ""
echo "=========================================="
echo "GPU Check:"
python -c "import torch; print('  CUDA available:', torch.cuda.is_available()); print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('  VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB') if torch.cuda.is_available() else None"
echo "=========================================="
echo ""
echo "Setup complete. Next: bash check_gpu.sh to verify the GPU is free before starting."
