# Troubleshooting Guide

Common issues and solutions for Lightweight Voting-LFQ training.

---

## Setup Issues

### CUDA Not Available

**Symptoms**:
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Solutions**:

1. **Verify NVIDIA Driver**:
```bash
nvidia-smi
# Should show GPU info, driver version, CUDA version
```

If `nvidia-smi` fails:
```bash
# Ubuntu
sudo apt install nvidia-driver-535
sudo reboot

# CentOS
sudo yum install nvidia-driver
sudo reboot
```

2. **Check PyTorch Installation**:
```bash
python -c "import torch; print(torch.version.cuda)"
# Should match your CUDA version (11.8 or 12.1)
```

If mismatch:
```bash
# Reinstall PyTorch with correct CUDA version
# For CUDA 11.8:
pip uninstall torch torchvision torchaudio
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

3. **Check CUDA_HOME**:
```bash
echo $CUDA_HOME
# Should be: /usr/local/cuda

# If not set:
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Add to ~/.bashrc for persistence
```

---

### Out of Memory (OOM) During Setup

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:

1. **Reduce Batch Size**:
```yaml
# Edit configs/training_config.yaml
training:
  batch_size: 8  # From 16
  val_batch_size: 16  # From 32
```

2. **Enable Gradient Checkpointing**:
```yaml
# Edit configs/training_config.yaml
training:
  use_gradient_checkpointing: true
```

3. **Reduce Number of Workers**:
```yaml
training:
  num_workers: 4  # From 8
```

4. **Clear GPU Cache**:
```python
import torch
torch.cuda.empty_cache()
```

5. **Use Single GPU**:
```bash
export CUDA_VISIBLE_DEVICES=0  # Use only GPU 0
```

---

### Dependency Installation Failures

**Symptoms**:
```
ERROR: Could not find a version that satisfies the requirement ...
```

**Solutions**:

1. **Update pip**:
```bash
pip install --upgrade pip setuptools wheel
```

2. **Install with --no-cache-dir**:
```bash
pip install --no-cache-dir -r requirements.txt
```

3. **Install dependencies individually**:
```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install datasets==2.14.0
pip install librosa==0.10.0
pip install onnx==1.14.0
pip install onnxruntime-gpu==1.15.0
pip install tensorboard==2.14.0
pip install pyyaml==6.0
pip install tqdm
```

---

## Data Download Issues

### Download Hangs or Fails

**Symptoms**:
```
Downloading: 0%|          | 0.00/1.24G [00:00<?, ?B/s]
[Connection timeout]
```

**Solutions**:

1. **Resume Download**:
```bash
bash scripts/download_data.sh --tier 1 --resume
```

2. **Use Direct HuggingFace Download**:
```bash
# Install huggingface-cli
pip install huggingface_hub[cli]

# Download manually
huggingface-cli download facebook/librispeech_asr \
  --repo-type dataset \
  --local-dir data/librispeech

huggingface-cli download mozilla-foundation/common_voice_11_0 \
  --repo-type dataset \
  --local-dir data/common_voice
```

3. **Check Disk Space**:
```bash
df -h
# Ensure >200GB free for Tier 2
```

4. **Increase Timeout**:
```bash
# Edit scripts/download_data.sh
# Add timeout parameter:
export HF_HUB_DOWNLOAD_TIMEOUT=3600  # 1 hour
```

---

### Corrupted Download

**Symptoms**:
```
python scripts/verify_data.py --tier 1
❌ LibriSpeech clean-100: Checksum mismatch
```

**Solutions**:

1. **Re-download Specific Dataset**:
```bash
rm -rf data/librispeech/train-clean-100
bash scripts/download_data.sh --dataset librispeech --split train-clean-100
```

2. **Verify Download Integrity**:
```bash
python scripts/verify_data.py --tier 1 --verbose
# Shows which files are corrupted
```

3. **Manual Download**:
```
Visit: https://www.openslr.org/12/
Download: train-clean-100.tar.gz
Extract to: data/librispeech/train-clean-100/
```

---

## Training Issues

### Loss Not Decreasing

**Symptoms**:
```
Epoch 1: loss=3.456
Epoch 2: loss=3.451
Epoch 3: loss=3.455
Epoch 4: loss=3.448
...
(No significant decrease)
```

**Diagnosis**:

1. **Check Zipformer Loading**:
```bash
grep "Loaded Zipformer" outputs/logs/training.log
# Should see: "Loaded Zipformer checkpoint: ..."
```

If missing:
```bash
bash scripts/download_zipformer.sh
python scripts/verify_zipformer.py
```

2. **Verify Frozen Encoder**:
```bash
grep "Frozen parameters" outputs/logs/training.log
# Should see: "Frozen Zipformer encoder: X parameters"
```

3. **Check Learning Rate**:
```bash
grep "Learning rate" outputs/logs/training.log
# Should start at 1e-4, not 0
```

**Solutions**:

1. **Reduce Learning Rate**:
```yaml
# configs/training_config.yaml
training:
  learning_rate: 5e-5  # From 1e-4
```

2. **Increase Warmup**:
```yaml
training:
  warmup_epochs: 10  # From 5
```

3. **Check Data Quality**:
```bash
python scripts/analyze_data.py --tier 1 --num-samples 100
# Shows audio duration distribution, transcript lengths
```

4. **Adjust Loss Weights**:
```yaml
losses:
  asr_weight: 1.5  # From 1.0 (prioritize ASR)
  mel_weight: 0.3  # From 0.5 (reduce TTS influence)
```

---

### Training Diverges (Loss → NaN)

**Symptoms**:
```
Epoch 5, Step 234: loss=2.345
Epoch 5, Step 235: loss=nan
```

**Solutions**:

1. **Enable Gradient Clipping**:
```yaml
# configs/training_config.yaml
training:
  grad_clip: 1.0  # Should already be set
```

2. **Reduce Learning Rate**:
```yaml
training:
  learning_rate: 5e-5  # From 1e-4
```

3. **Check for Inf/NaN in Data**:
```bash
python scripts/check_data_quality.py --tier 1
```

4. **Use FP32 Instead of FP16**:
```yaml
training:
  use_amp: false  # Disable mixed precision
```

5. **Reduce Batch Size** (helps with numerical stability):
```yaml
training:
  batch_size: 8  # From 16
```

---

### High WER (>20%)

**Symptoms**:
```
Epoch 50: Validation WER = 24.5%
(Expected: <10%)
```

**Solutions**:

1. **Train Longer**:
```bash
python scripts/train.py \
  --config configs/training_config.yaml \
  --tier 1 \
  --num-epochs 75  # From 50
```

2. **Increase ASR Loss Weight**:
```yaml
losses:
  asr_weight: 1.5  # From 1.0
```

3. **Reduce Mel Loss Weight** (focus on ASR):
```yaml
losses:
  mel_weight: 0.2  # From 0.5
```

4. **Check Adapter Output**:
```bash
python scripts/debug_model.py \
  --checkpoint outputs/tier1_run1/checkpoints/latest_model.pt \
  --check-adapter-output
# Shows MSE between adapter output and encoder features
# Target: <0.5
```

5. **Verify Zipformer Decoder Not Frozen**:
```bash
# Actually, decoder SHOULD be frozen
# If you accidentally unfroze it:
grep "decoder.*requires_grad" outputs/logs/training.log
# Should all be False
```

---

### Low Consensus Score (<0.6)

**Symptoms**:
```
Epoch 50: Consensus = 0.52
(Expected: >0.8)
```

**Solutions**:

1. **Increase Consensus Weight**:
```yaml
losses:
  consensus_weight: 0.2  # From 0.1
```

2. **Increase Commitment Weight**:
```yaml
losses:
  commitment_weight: 0.5  # From 0.25
```

3. **Reduce Codebook Size** (easier to agree):
```yaml
# configs/model_config.yaml
voting_lfq:
  codebook_size: 2048  # From 4096
  # Note: This requires retraining from scratch
```

4. **Train Longer**:
- Consensus often improves in later epochs
- Check tensorboard: `train/consensus_score` trend

5. **Check Branch Diversity**:
```bash
python scripts/analyze_branches.py \
  --checkpoint outputs/tier1_run1/checkpoints/latest_model.pt
# Shows per-branch statistics
```

---

### High Mel Reconstruction Loss (>0.8)

**Symptoms**:
```
Epoch 50: Mel L1 = 1.2
(Expected: <0.5)
```

**Impact**: TTS quality will be poor

**Solutions**:

1. **Increase Mel Weight**:
```yaml
losses:
  mel_weight: 0.8  # From 0.5
```

2. **Increase Embedding Dimension** (more capacity):
```yaml
# configs/model_config.yaml
voting_lfq:
  embedding_dim: 512  # From 256
  # Note: Increases model size
```

3. **Train Mel Decoder Separately First**:
```bash
python scripts/pretrain_mel_decoder.py \
  --config configs/training_config.yaml \
  --epochs 20
# Then resume full training
```

4. **Check Mel Feature Extraction**:
```bash
python scripts/verify_mel_extraction.py \
  --audio-file data/librispeech/sample.wav
# Verifies mel spectrogram computation
```

---

## Distributed Training Issues

### DDP Hangs at Initialization

**Symptoms**:
```
Initializing distributed training...
[Hangs indefinitely]
```

**Solutions**:

1. **Check Port Availability**:
```bash
# Default port: 29500
netstat -tuln | grep 29500

# If in use, change port:
export MASTER_PORT=29501
```

2. **Set Explicit Backend**:
```bash
export NCCL_SOCKET_IFNAME=eth0  # Or your network interface
export NCCL_DEBUG=INFO  # Verbose NCCL logging
```

3. **Verify GPU Visibility**:
```bash
python -c "import torch; print(torch.cuda.device_count())"
# Should show >1 for multi-GPU
```

4. **Use Single GPU to Debug**:
```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/train.py --config configs/training_config.yaml --tier 1
```

---

### Uneven GPU Utilization

**Symptoms**:
```
GPU 0: 95% utilization
GPU 1: 45% utilization
```

**Solutions**:

1. **Check Batch Distribution**:
```yaml
# Ensure batch_size is divisible by num_GPUs
# 2 GPUs, batch_size should be 16, 32, etc.
training:
  batch_size: 16  # 8 per GPU
```

2. **Balance Data Loading**:
```yaml
training:
  num_workers: 8  # Multiple of num_GPUs
```

3. **Check for Blocking Operations**:
```bash
# Look for synchronization points in code
grep "dist.barrier" src/training/*.py
```

---

## ONNX Export Issues

### ONNX Export Fails

**Symptoms**:
```
RuntimeError: Exporting the operator voting_lfq to ONNX opset version 14 is not supported.
```

**Solutions**:

1. **Check PyTorch Version**:
```bash
python -c "import torch; print(torch.__version__)"
# Should be 2.1.0+
```

2. **Simplify Model for Export**:
```bash
python scripts/export_onnx.py \
  --checkpoint outputs/tier1_run1/checkpoints/best_model.pt \
  --output-dir outputs/onnx_models \
  --opset-version 14 \  # Try different versions: 13, 14, 15
  --simplify  # Enable ONNX simplification
```

3. **Export Components Separately**:
```bash
# If full model export fails, export individually:
python scripts/export_onnx.py \
  --checkpoint outputs/tier1_run1/checkpoints/best_model.pt \
  --export-only voting_lfq

python scripts/export_onnx.py \
  --checkpoint outputs/tier1_run1/checkpoints/best_model.pt \
  --export-only embedding

python scripts/export_onnx.py \
  --checkpoint outputs/tier1_run1/checkpoints/best_model.pt \
  --export-only adapter
```

4. **Check for Unsupported Operations**:
```bash
python scripts/check_onnx_ops.py \
  --checkpoint outputs/tier1_run1/checkpoints/best_model.pt
# Lists all operations, highlights unsupported
```

---

### ONNX Model Produces Wrong Results

**Symptoms**:
```
PyTorch output: [10, 234, 56, 789, ...]
ONNX output:    [10, 234, 56, 892, ...]
```

**Solutions**:

1. **Validate ONNX Export**:
```bash
python scripts/validate_onnx.py \
  --pytorch-checkpoint outputs/tier1_run1/checkpoints/best_model.pt \
  --onnx-dir outputs/onnx_models \
  --num-samples 100 \  # Test on 100 samples
  --tolerance 1e-5  # Acceptable difference
```

2. **Check Quantization**:
```bash
# If using INT8 quantization:
python scripts/export_onnx.py \
  --checkpoint outputs/tier1_run1/checkpoints/best_model.pt \
  --output-dir outputs/onnx_models_fp32 \
  --quantize false  # Export FP32 first

# Then validate FP32 matches PyTorch
# Then quantize if needed
```

3. **Verify ONNX Runtime Version**:
```bash
python -c "import onnxruntime; print(onnxruntime.__version__)"
# Should be 1.15.0+
```

4. **Check for Dynamic Shapes**:
```bash
# Ensure input shapes are correct
python scripts/test_onnx_shapes.py \
  --onnx-model outputs/onnx_models/voting_lfq.onnx \
  --input-shape 1 100 512  # [Batch, Time, Features]
```

---

## Performance Issues

### Slow Training (< 5 samples/sec)

**Expected**: 10-15 samples/sec on single GPU

**Solutions**:

1. **Enable Mixed Precision**:
```yaml
training:
  use_amp: true
```

2. **Increase Batch Size**:
```yaml
training:
  batch_size: 32  # From 16 (if GPU memory allows)
```

3. **Optimize Data Loading**:
```yaml
training:
  num_workers: 8  # From 4
  prefetch_factor: 2
  pin_memory: true
```

4. **Profile Training**:
```bash
python scripts/profile_training.py \
  --config configs/training_config.yaml \
  --tier 1 \
  --max-steps 100
# Shows time breakdown: data loading, forward, backward, optimizer
```

5. **Check CPU Bottleneck**:
```bash
htop  # Monitor CPU usage
# If CPU at 100%, data loading is bottleneck → increase num_workers
```

---

### Slow Validation (> 10 minutes)

**Solutions**:

1. **Reduce Validation Samples**:
```yaml
training:
  val_max_samples: 1000  # Validate on subset
```

2. **Increase Validation Batch Size**:
```yaml
training:
  val_batch_size: 64  # From 32
```

3. **Skip Validation During Training**:
```bash
python scripts/train.py \
  --config configs/training_config.yaml \
  --tier 1 \
  --validate-every 5  # Validate every 5 epochs instead of 1
```

---

## System Issues

### Disk Space Running Out

**Symptoms**:
```
OSError: [Errno 28] No space left on device
```

**Solutions**:

1. **Clean Old Checkpoints**:
```bash
python scripts/clean_checkpoints.py \
  --output-dir outputs/tier1_run1 \
  --keep-best \
  --keep-latest \
  --keep-last-n 2  # Keep only 2 recent checkpoints
```

2. **Move Data to Separate Drive**:
```bash
# Move datasets
mv data /mnt/external_drive/data
ln -s /mnt/external_drive/data data

# Update config
# Edit configs/data_config.yaml:
# data_root: /mnt/external_drive/data
```

3. **Clean Cache**:
```bash
rm -rf cache/*  # Remove preprocessed features cache
rm -rf ~/.cache/huggingface  # Remove HF cache
```

---

### Network Issues (Multi-GPU)

**Symptoms**:
```
NCCL error: unhandled system error
```

**Solutions**:

1. **Check Network Interface**:
```bash
ifconfig
# Find correct interface (e.g., eth0, ib0)

export NCCL_SOCKET_IFNAME=eth0  # Or ib0 for InfiniBand
```

2. **Disable NCCL P2P** (if issues persist):
```bash
export NCCL_P2P_DISABLE=1
```

3. **Use Gloo Backend** (slower but more stable):
```bash
export TORCH_DISTRIBUTED_BACKEND=gloo
```

---

## Getting Help

If issues persist:

1. **Collect Diagnostic Information**:
```bash
bash scripts/collect_diagnostics.sh > diagnostics.txt
# Includes: GPU info, CUDA version, PyTorch version, logs
```

2. **Check Logs**:
```bash
# Training log
cat outputs/tier1_run1/logs/training.log

# Error log
cat outputs/tier1_run1/logs/error.log
```

3. **Enable Debug Mode**:
```bash
python scripts/train.py \
  --config configs/training_config.yaml \
  --tier 1 \
  --debug \
  --verbose
```

4. **Test Individual Components**:
```bash
# Test Voting-LFQ
python tests/test_voting_lfq.py

# Test data loading
python tests/test_data_loader.py

# Test training step
python tests/test_training_step.py
```

---

## Quick Checklist

Before opening an issue, verify:

- [ ] `nvidia-smi` shows GPUs
- [ ] `python scripts/verify_setup.py` passes
- [ ] `python scripts/verify_data.py --tier X` passes
- [ ] Zipformer checkpoint downloaded
- [ ] Sufficient disk space (>100GB free)
- [ ] Logs show training started (not hanging)
- [ ] No NaN/Inf in losses (check tensorboard)

---

## Emergency Recovery

**Training Crashed Midway**:
```bash
# Resume from latest checkpoint
python scripts/train.py \
  --config configs/training_config.yaml \
  --tier 1 \
  --resume outputs/tier1_run1/checkpoints/latest_model.pt
```

**Corrupted Checkpoint**:
```bash
# Resume from previous epoch
python scripts/train.py \
  --config configs/training_config.yaml \
  --tier 1 \
  --resume outputs/tier1_run1/checkpoints/epoch_45.pt
```

**Complete Reset**:
```bash
# Start from scratch
rm -rf outputs/tier1_run1
python scripts/train.py \
  --config configs/training_config.yaml \
  --tier 1 \
  --output-dir outputs/tier1_run2
```
