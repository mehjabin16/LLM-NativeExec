# Running LLMs Natively on Different Backends 
This covers how to run the native.py script with different hardware setups: CPU, CUDA (GPU), and MPS (Mac).
The native_quant.py has additional option for 4-bit and 8-bit quantization using BitsAndBytesConfig, as 4-bit quantization significantly reduces the memory footprint of large language models.

# General Command Format:
```
python native.py 
--models <model_name>
--json_files <file.json> 
--device <device_type> 
--output_dir <output_dir>
```

# Hugging Face Token:
```
from huggingface_hub import login
login(token="your_huggingface_token_here")
```
--------------------------------------------------
# 💻 CPU Setup (All Platforms):
Requirements:
- create virtual environment

```
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```
- install dependencies
```
 pip install -r requirements.txt
```

Command:
```
python native.py ^
  --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 ^
  --json_files alpaca_001.json ^
  --device cpu ^
  --output_dir results
```

* Use `^` for line breaks in Windows CMD/PowerShell. Use `\` on macOS/Linux.

--------------------------------------------------
# 🚀 CUDA Setup (NVIDIA GPU on Windows/Linux):
1. Install PyTorch with CUDA support:
```
 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
2. Ensure you have a compatible GPU and CUDA drivers installed.

3. Set device as `cuda` in the command:

Command:
```
python native.py ^
  --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 ^
  --json_files alpaca_001.json ^
  --device cuda ^
  --output_dir results
```

```
#Quantized Version#
python native_quant.py ^
  --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 ^
  --json_files alpaca_001.json ^
  --device auto ^
  --quantization 4bit ^
  --output_dir results
```
Note:
If the module bitsandbytes not found - 
pip install git+https://github.com/TimDettmers/bitsandbytes.git 

--------------------------------------------------
#🍎 MPS Setup (Mac with Apple Silicon):#
1. Install PyTorch with MPS support (macOS):
```
   pip install torch torchvision torchaudio
```
2. Set device as `mps` in the command:

Command:
```
python native.py \
  --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --json_files alpaca_001.json \
  --device mps \
  --output_dir results
```

 MPS is still experimental and may be slower than expected.
```
##Quantized Version##
python native_quant.py \
  --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --json_files alpaca_001.json \ 
  --device auto \
  --quantization 4bit \
  --output_dir results
  ```
--------------------------------------------------
💡 Notes:
- Always check if your device supports CUDA or MPS with:
  ```python
  import torch
  print(torch.cuda.is_available())   # for CUDA
  print(torch.backends.mps.is_available())  # for MPS
