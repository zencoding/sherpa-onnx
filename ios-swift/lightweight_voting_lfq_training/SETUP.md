# Setup Guide - Lightweight Voting-LFQ Training Package

**Estimated Time**: 30 minutes
**Target Environment**: Linux with NVIDIA GPU
**Prerequisites**: Root/sudo access, internet connection

---

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 16GB+ VRAM (e.g., RTX 3090, A100, V100)
- **RAM**: 32GB+ system RAM recommended
- **Storage**: 200GB+ free disk space
  - Training data: ~150GB (Tier 2) or ~25GB (Tier 1)
  - Model checkpoints: ~10GB
  - Working space: ~20GB
- **CPU**: 8+ cores recommended for data loading

### Software
- **OS**: Ubuntu 20.04+ or CentOS 7+ (other Linux distributions should work)
- **CUDA**: 11.8 or 12.1+ with compatible cuDNN
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **Git**: For cloning repositories (optional but recommended)

---

## Quick Start Installation

### Step 1: Extract Package

```bash
# Transfer completed, now extract
tar -xzf voting_lfq_training.tar.gz
cd lightweight_voting_lfq_training

# Verify extraction
ls -la
# Should see: configs/ src/ scripts/ docs/ README.md etc.
```

### Step 2: Verify CUDA Installation

```bash
# Check CUDA version
nvidia-smi

# Expected output should show:
# - Driver version
# - CUDA version (11.8+ or 12.1+)
# - GPU model and memory

# If nvidia-smi fails, install NVIDIA drivers:
# Ubuntu: sudo apt install nvidia-driver-535 nvidia-cuda-toolkit
# CentOS: sudo yum install nvidia-driver nvidia-cuda-toolkit
```

### Step 3: Create Python Environment

**Option A: Using conda (recommended)**

```bash
# Install conda if not available
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Create environment
conda create -n voting_lfq python=3.10 -y
conda activate voting_lfq
```

**Option B: Using venv**

```bash
# Ensure Python 3.8+ is installed
python3 --version

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
# For CUDA 11.8:
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install package requirements
pip install -r requirements.txt

# This installs:
# - datasets (HuggingFace)
# - librosa (audio processing)
# - onnx, onnxruntime (model export)
# - tensorboard (monitoring)
# - pyyaml (configuration)
# - tqdm (progress bars)
# - numpy, scipy
```

### Step 5: Verify Installation

```bash
# Run verification script
python scripts/verify_setup.py

# Expected output:
# ✅ Python version: 3.10.x
# ✅ PyTorch version: 2.1.0
# ✅ CUDA available: True
# ✅ CUDA version: 11.8 (or 12.1)
# ✅ GPU count: 1 (or more)
# ✅ GPU 0: NVIDIA RTX 3090 (24GB)
# ✅ All dependencies installed
# ✅ Package structure verified
```

---

## Detailed Setup

### CUDA and cuDNN Installation

**Ubuntu 20.04/22.04**

```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update

# Install CUDA toolkit
sudo apt install cuda-toolkit-11-8 -y

# Install cuDNN
sudo apt install libcudnn8 libcudnn8-dev -y

# Add to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**CentOS 7/8**

```bash
# Add NVIDIA repository
sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# Install CUDA toolkit
sudo yum install cuda-toolkit-11-8 -y

# Install cuDNN (requires NVIDIA Developer account)
# Download cuDNN from https://developer.nvidia.com/cudnn
sudo rpm -i cudnn*.rpm

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Multi-GPU Configuration

If you have multiple GPUs, the training script will automatically detect and use them.

```bash
# Check available GPUs
nvidia-smi --list-gpus

# Set specific GPUs (optional)
export CUDA_VISIBLE_DEVICES=0,1  # Use GPU 0 and 1 only

# Training will automatically use all visible GPUs with DistributedDataParallel
```

### Storage Configuration

```bash
# Check available disk space
df -h

# Recommended directory structure:
# /workspace/lightweight_voting_lfq_training  (~10MB, code)
# /data/audio_datasets                        (~150GB, datasets)
# /workspace/outputs                          (~10GB, checkpoints)

# If storage is limited, use separate data directory
mkdir -p /data/audio_datasets

# Update configs/data_config.yaml:
# data_root: /data/audio_datasets
```

---

## Data Download

### Tier 1: Quick Validation (100 hours)

**Recommended for first run to validate setup.**

```bash
# Download LibriSpeech clean-100
bash scripts/download_data.sh --tier 1

# Download time: 1-2 hours (depends on bandwidth)
# Disk usage: ~25GB

# Verify download
python scripts/verify_data.py --tier 1

# Expected output:
# ✅ LibriSpeech clean-100: 28539 samples, 100.6 hours
# ✅ Audio files accessible
# ✅ Transcripts loaded
```

### Tier 2: Production Quality (1460 hours)

**Use after Tier 1 validation succeeds.**

```bash
# Download full training data
bash scripts/download_data.sh --tier 2

# Download time: 6-8 hours
# Disk usage: ~150GB

# Datasets downloaded:
# - LibriSpeech: train-clean-100, train-clean-360, train-other-500
# - Common Voice: en subset (~500 hours)

# Verify download
python scripts/verify_data.py --tier 2

# Expected output:
# ✅ LibriSpeech total: 281,241 samples, 960.3 hours
# ✅ Common Voice: 492,712 samples, 500.1 hours
# ✅ Total training data: 773,953 samples, 1460.4 hours
```

### Download Options

```bash
# Resume interrupted download
bash scripts/download_data.sh --tier 2 --resume

# Specify custom data directory
bash scripts/download_data.sh --tier 1 --data-dir /data/audio_datasets

# Download only specific dataset
bash scripts/download_data.sh --dataset librispeech
bash scripts/download_data.sh --dataset common_voice
```

---

## Zipformer Checkpoint Download

The Lightweight Voting-LFQ module is inserted into an existing Zipformer encoder. You need the pre-trained Zipformer checkpoint.

```bash
# Download Zipformer checkpoint
bash scripts/download_zipformer.sh

# This downloads:
# - Encoder weights (bilingual zh-en model)
# - Decoder weights
# - Configuration files
# Total size: ~330MB

# Checkpoint saved to:
# pretrained_models/zipformer_bilingual/
#   ├── encoder.pt
#   ├── decoder.pt
#   ├── joiner.pt
#   └── config.yaml

# Verify checkpoint
python scripts/verify_zipformer.py

# Expected output:
# ✅ Encoder loaded: 512-dim output
# ✅ Decoder loaded: compatible
# ✅ Test forward pass successful
```

---

## Configuration Review

Before starting training, review and customize configurations:

### configs/model_config.yaml

```yaml
# Voting-LFQ settings
voting_lfq:
  num_branches: 3              # Voting branches
  embedding_dim: 256           # Token embedding size
  codebook_size: 4096          # Vocabulary size (2^12)
  commitment_weight: 0.25      # Commitment loss weight

# Adapter settings
adapter:
  input_dim: 256               # From embeddings
  output_dim: 512              # Match Zipformer decoder input

# Review and adjust if needed
```

### configs/training_config.yaml

```yaml
# Training settings
training:
  batch_size: 16               # Adjust based on GPU memory
  learning_rate: 1e-4          # Starting LR
  num_epochs: 50               # Tier 1: 50, Tier 2: 100

  # Distributed training
  use_ddp: true                # Auto-detect multiple GPUs

  # Mixed precision
  use_amp: true                # FP16 training (2× faster)

# Adjust batch_size if you see OOM errors:
# - 24GB GPU: batch_size 16
# - 16GB GPU: batch_size 8
# - 12GB GPU: batch_size 4
```

### configs/data_config.yaml

```yaml
# Data paths (update if using custom directories)
data:
  data_root: ./data            # Default: ./data
  cache_dir: ./cache           # Preprocessed features

# Adjust if needed:
# data_root: /data/audio_datasets
```

---

## Quick Test Run

Before starting full training, run a quick test to verify everything works:

```bash
# Test with tiny subset (10 samples, 5 steps)
python scripts/train.py \
  --config configs/training_config.yaml \
  --tier 1 \
  --debug \
  --max-samples 10 \
  --max-steps 5

# Expected output:
# Loading Zipformer checkpoint...
# Creating Voting-LFQ module...
# Loading training data: 10 samples
# Epoch 1/1, Step 1/5: loss=2.456
# Epoch 1/1, Step 2/5: loss=2.301
# ...
# ✅ Debug run completed successfully

# If this succeeds, you're ready for full training!
```

---

## Troubleshooting Setup Issues

### CUDA not available

```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False:
# 1. Verify nvidia-smi works
# 2. Reinstall PyTorch with correct CUDA version
# 3. Check CUDA_HOME environment variable
export CUDA_HOME=/usr/local/cuda
```

### Out of Memory (OOM)

```bash
# Reduce batch size in configs/training_config.yaml
# From: batch_size: 16
# To:   batch_size: 8 (or 4)

# Enable gradient checkpointing
# In configs/training_config.yaml:
# use_gradient_checkpointing: true
```

### Download failures

```bash
# Use resume flag
bash scripts/download_data.sh --tier 1 --resume

# Or manually download from HuggingFace:
# https://huggingface.co/datasets/librispeech_asr
# https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0

# Extract to: data/librispeech, data/common_voice
```

### Import errors

```bash
# Reinstall requirements
pip install --force-reinstall -r requirements.txt

# Check for missing packages
python scripts/verify_setup.py
```

### Permission denied

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Or run with bash explicitly
bash scripts/download_data.sh --tier 1
```

---

## Next Steps

Once setup is complete:

1. **Start Training**: See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
2. **Monitor Progress**: TensorBoard and logs
3. **Export Models**: After training completes

---

## System Resource Monitoring

```bash
# Monitor GPU usage during training
watch -n 1 nvidia-smi

# Monitor disk usage
watch -n 60 df -h

# Monitor CPU and RAM
htop

# Check training process
ps aux | grep python
```

---

## Backup and Restore

```bash
# Backup checkpoints (recommended every 24 hours)
tar -czf outputs_backup_$(date +%Y%m%d).tar.gz outputs/

# Copy to safe location
scp outputs_backup_*.tar.gz user@backup-server:/backups/

# Restore from backup
tar -xzf outputs_backup_20251003.tar.gz
```

---

## Summary

**Setup Checklist**:
- ✅ Extract package
- ✅ Verify CUDA with `nvidia-smi`
- ✅ Create Python environment
- ✅ Install PyTorch + dependencies
- ✅ Run `python scripts/verify_setup.py`
- ✅ Download training data (Tier 1 or Tier 2)
- ✅ Download Zipformer checkpoint
- ✅ Review configurations
- ✅ Run debug test
- ✅ Ready to train!

**Estimated Total Time**: 30 minutes (code) + 1-8 hours (data download)

**Ready to proceed?** → [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
