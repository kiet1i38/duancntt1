#!/usr/bin/env bash
set -euo pipefail
python -m pip install 'torch==2.3.0' --index-url https://download.pytorch.org/whl/cu121
python -m pip install 'numpy==1.26.4' 'transformers==4.47.0' 'peft==0.14.0' \
  'accelerate==1.2.1' 'sympy==1.13.3' 'regex==2024.11.6' \
  'antlr4-python3-runtime==4.11.1' 'sentencepiece==0.2.0' 'safetensors==0.4.5' \
  'tokenizers==0.21.4' 'huggingface-hub==0.27.1'
# The base image may contain a torchvision build tied to a different torch.
# Text-only inference does not need it, and its import otherwise breaks PEFT.
python -m pip uninstall -y torchvision
python -c "import torch, transformers, peft; assert torch.cuda.is_available(); print(torch.__version__, transformers.__version__, peft.__version__, torch.cuda.get_device_name(0))"
