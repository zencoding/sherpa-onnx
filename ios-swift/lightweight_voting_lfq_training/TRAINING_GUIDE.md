# Training Guide - Lightweight Voting-LFQ

**For**: Training the noise-robust semantic tokenizer module
**Input**: Audio + transcripts (HuggingFace datasets)
**Output**: ONNX models ready for iOS deployment

---

## Training Overview

### What Gets Trained

```
┌─────────────────────────────────────────────────┐
│            FROZEN COMPONENTS                    │
│  (Use existing Zipformer weights, no training)  │
├─────────────────────────────────────────────────┤
│  • Zipformer Encoder (bilingual zh-en)          │
│  • Zipformer Decoder (transducer decoder)       │
│  • Zipformer Joiner (prediction network)        │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         TRAINABLE COMPONENTS (~6MB)             │
├─────────────────────────────────────────────────┤
│  1. Voting-LFQ Module (~5MB)                    │
│     - 3 parallel quantization branches          │
│     - Shared base projection                    │
│     - Bit-wise majority voting logic            │
│                                                 │
│  2. Token Embedding Layer (~1MB)                │
│     - 4096 × 256 embedding matrix               │
│                                                 │
│  3. Adapter Projection (~0.5MB)                 │
│     - 256 → 512 dimension projection            │
│                                                 │
│  4. Mel Decoder (TTS validation, ~0.5MB)        │
│     - Validates acoustic preservation           │
│     - NOT exported to iOS                       │
└─────────────────────────────────────────────────┘
```

### Training Flow

```
Audio (16kHz) → Zipformer Encoder (frozen)
                     ↓
              Hidden States [B, T, 512]
                     ↓
           Voting-LFQ (trainable)
           - Projects to branches
           - Quantizes with voting
                     ↓
              Semantic Tokens [B, T]
                     ↓
           Embedding (trainable)
                     ↓
           Adapter (trainable)
                     ↓
              Reconstructed Features [B, T, 512]
                     ↓
           ┌────────┴────────┐
           ↓                 ↓
    Zipformer Decoder    Mel Decoder
    (frozen, ASR)        (trainable, TTS)
           ↓                 ↓
    Text Output         Mel Spectrogram
    (compute WER)       (compute L1 loss)
```

---

## Starting Training

### Tier 1: Quick Validation (1-2 days)

**Purpose**: Validate implementation, debug issues, test pipeline
**Data**: LibriSpeech clean-100 (100 hours)
**GPU**: 1-2× GPUs with 16GB+ VRAM
**Time**: 24-48 hours

```bash
# Ensure setup completed
python scripts/verify_setup.py
python scripts/verify_data.py --tier 1

# Start Tier 1 training
python scripts/train.py \
  --config configs/training_config.yaml \
  --tier 1 \
  --output-dir outputs/tier1_run1

# Training will:
# 1. Load Zipformer checkpoint (frozen)
# 2. Initialize Voting-LFQ + Embedding + Adapter
# 3. Load LibriSpeech clean-100
# 4. Train for 50 epochs (~24-48 hours)
# 5. Save checkpoints every 5 epochs
# 6. Validate on dev set every epoch
```

### Tier 2: Production Quality (3-5 days)

**Purpose**: Production-ready model for iOS deployment
**Data**: LibriSpeech (960h) + Common Voice (500h)
**GPU**: 2-4× GPUs with 16GB+ VRAM
**Time**: 72-120 hours

```bash
# After Tier 1 succeeds, download Tier 2 data
bash scripts/download_data.sh --tier 2
python scripts/verify_data.py --tier 2

# Start Tier 2 training
python scripts/train.py \
  --config configs/training_config.yaml \
  --tier 2 \
  --output-dir outputs/tier2_production

# Training will:
# 1. Load best Tier 1 checkpoint (optional warm start)
# 2. Train on 1460 hours of audio
# 3. Run for 100 epochs (~3-5 days)
# 4. Validate on both LibriSpeech and Common Voice dev sets
```

---

## Training Command Options

### Basic Usage

```bash
python scripts/train.py \
  --config configs/training_config.yaml \
  --tier [1|2] \
  --output-dir outputs/my_run
```

### Advanced Options

```bash
python scripts/train.py \
  --config configs/training_config.yaml \
  --tier 2 \
  --output-dir outputs/tier2_run1 \
  --resume outputs/tier1_run1/checkpoints/best_model.pt \  # Warm start
  --num-epochs 100 \                                        # Override config
  --batch-size 16 \                                         # Override config
  --learning-rate 1e-4 \                                    # Override config
  --gpu-ids 0,1,2,3 \                                       # Use specific GPUs
  --num-workers 8 \                                         # Data loading threads
  --validate-every 1 \                                      # Validate every N epochs
  --save-every 5 \                                          # Checkpoint every N epochs
  --tensorboard-dir outputs/tier2_run1/tensorboard         # TensorBoard logs
```

### Debug Mode

```bash
# Quick test with minimal data
python scripts/train.py \
  --config configs/training_config.yaml \
  --tier 1 \
  --debug \
  --max-samples 100 \      # Use only 100 samples
  --max-steps 50 \         # Train for 50 steps only
  --validate-every 10      # Validate every 10 steps
```

---

## Monitoring Training

### Real-Time Logs

```bash
# View training logs in real-time
tail -f outputs/tier1_run1/logs/training.log

# Expected output:
# [2025-10-03 10:15:23] Epoch 1/50, Step 100/6250
# [2025-10-03 10:15:23] Loss: 2.345 | ASR: 1.234 | Consensus: 0.678 | Commitment: 0.234 | Mel: 0.199
# [2025-10-03 10:15:23] LR: 0.0001 | GPU Mem: 14.2GB / 24.0GB
# [2025-10-03 10:15:23] Throughput: 12.3 samples/sec
```

### TensorBoard

```bash
# On training machine
tensorboard --logdir outputs/tier1_run1/tensorboard --port 6006

# Access via SSH tunnel from dev machine
ssh -L 6006:localhost:6006 user@training-machine

# Open browser: http://localhost:6006

# Metrics to monitor:
# - train/total_loss (should decrease steadily)
# - train/asr_loss (target <1.0)
# - train/consensus_score (target >0.8)
# - train/mel_reconstruction_loss (target <0.5)
# - val/wer (target <10% on LibriSpeech clean)
# - learning_rate (follows cosine schedule)
```

### Checkpoint Files

```bash
# Checkpoints saved to:
outputs/tier1_run1/checkpoints/
├── epoch_5.pt          # Checkpoint at epoch 5
├── epoch_10.pt         # Checkpoint at epoch 10
├── best_model.pt       # Best validation WER
└── latest_model.pt     # Latest checkpoint (resume)

# Each checkpoint contains:
# - voting_lfq: Module weights
# - embedding: Embedding layer weights
# - adapter: Adapter weights
# - mel_decoder: Mel decoder weights (not exported)
# - optimizer: Optimizer state
# - scheduler: LR scheduler state
# - epoch: Current epoch number
# - best_wer: Best validation WER seen
```

---

## Training Hyperparameters

### configs/training_config.yaml

```yaml
training:
  # Optimization
  batch_size: 16                 # Samples per GPU
  learning_rate: 1e-4            # Initial LR
  weight_decay: 0.01             # L2 regularization
  num_epochs: 50                 # Tier 1: 50, Tier 2: 100

  # Learning rate schedule
  lr_scheduler: cosine           # Cosine annealing
  warmup_epochs: 5               # Linear warmup
  min_lr: 1e-6                   # Minimum LR

  # Optimization
  optimizer: adamw               # AdamW optimizer
  grad_clip: 1.0                 # Gradient clipping

  # Distributed training
  use_ddp: true                  # Auto-enable if multiple GPUs
  find_unused_parameters: false  # DDP optimization

  # Mixed precision
  use_amp: true                  # FP16 training (2× faster)

  # Data loading
  num_workers: 8                 # Parallel data loading
  prefetch_factor: 2             # Samples to prefetch

  # Validation
  validate_every: 1              # Epochs between validation
  val_batch_size: 32             # Larger for faster validation

  # Checkpointing
  save_every: 5                  # Save checkpoint every N epochs
  keep_last_n: 3                 # Keep last 3 checkpoints
  save_best: true                # Save best WER checkpoint
```

### Loss Weights

```yaml
losses:
  # Multi-task training weights
  asr_weight: 1.0                # ASR reconstruction loss
  consensus_weight: 0.1          # Branch agreement loss
  commitment_weight: 0.25        # Codebook commitment
  mel_weight: 0.5                # TTS compatibility

  # ASR loss
  asr_loss_type: mse             # MSE for feature reconstruction

  # Consensus loss (voting branch agreement)
  consensus_threshold: 0.8       # Target agreement score

  # Mel reconstruction (TTS validation)
  mel_loss_type: l1              # L1 loss for mel spectrogram
```

**Adjusting Weights**:
- **High ASR focus**: `asr_weight: 1.5`, `mel_weight: 0.3`
- **High TTS focus**: `asr_weight: 0.8`, `mel_weight: 0.8`
- **Balanced (recommended)**: Keep default values

---

## Training Phases Explained

### Phase 1: Warmup (Epochs 1-5)

**Goal**: Stabilize initial training, prevent gradient explosions

```
Learning Rate: 0 → 1e-4 (linear increase)
Behavior:
- Voting-LFQ learns basic quantization patterns
- Embedding layer initializes token representations
- Consensus score improves: 0.2 → 0.5
Expected Metrics:
- Total loss: 3.5 → 2.8
- ASR loss: 2.0 → 1.5
- Consensus: 0.2 → 0.5
```

### Phase 2: Rapid Improvement (Epochs 6-20)

**Goal**: Fast convergence to reasonable performance

```
Learning Rate: 1e-4 (constant)
Behavior:
- Quantization codebook specialization
- Strong acoustic pattern learning
- Consensus score: 0.5 → 0.75
Expected Metrics:
- Total loss: 2.8 → 1.8
- ASR loss: 1.5 → 0.8
- Validation WER: 25% → 12%
- Consensus: 0.5 → 0.75
```

### Phase 3: Fine-Tuning (Epochs 21-50)

**Goal**: Refinement and noise robustness

```
Learning Rate: 1e-4 → 1e-6 (cosine decay)
Behavior:
- Refinement of token boundaries
- Noise robustness improvement
- Consensus score: 0.75 → 0.85
Expected Metrics:
- Total loss: 1.8 → 1.2
- ASR loss: 0.8 → 0.5
- Validation WER: 12% → 8%
- Consensus: 0.75 → 0.85
- Mel reconstruction: 0.8 → 0.4
```

---

## Validation Metrics

### Primary Metrics

**1. Word Error Rate (WER)**
- **Target**: <10% on LibriSpeech test-clean
- **Calculation**: (Substitutions + Deletions + Insertions) / Total Words
- **Interpretation**:
  - WER < 5%: Excellent
  - WER 5-10%: Good (production-ready)
  - WER 10-15%: Acceptable (needs tuning)
  - WER > 15%: Poor (investigate issues)

**2. Consensus Score**
- **Target**: >0.8 (80% branch agreement)
- **Calculation**: Fraction of bits where all 3 branches agree
- **Interpretation**:
  - >0.85: Excellent voting stability
  - 0.75-0.85: Good (target range)
  - 0.65-0.75: Acceptable
  - <0.65: Poor (may need more training)

**3. Mel Reconstruction Loss (L1)**
- **Target**: <0.5 (TTS compatibility)
- **Calculation**: Mean absolute error between original and reconstructed mel spectrograms
- **Interpretation**:
  - <0.3: Excellent acoustic preservation
  - 0.3-0.5: Good (TTS-ready)
  - 0.5-0.8: Acceptable
  - >0.8: Poor (TTS may not work well)

### Secondary Metrics

**ASR Reconstruction Loss (MSE)**
- Target: <0.5
- Measures how well tokens preserve ASR features

**Commitment Loss**
- Target: <0.3
- Measures embedding stability

---

## Resuming Training

### From Checkpoint

```bash
# Resume from latest checkpoint
python scripts/train.py \
  --config configs/training_config.yaml \
  --tier 1 \
  --resume outputs/tier1_run1/checkpoints/latest_model.pt

# Resume from specific epoch
python scripts/train.py \
  --config configs/training_config.yaml \
  --tier 1 \
  --resume outputs/tier1_run1/checkpoints/epoch_25.pt
```

### Warm Start from Tier 1

```bash
# Start Tier 2 with Tier 1 weights
python scripts/train.py \
  --config configs/training_config.yaml \
  --tier 2 \
  --resume outputs/tier1_run1/checkpoints/best_model.pt \
  --reset-optimizer  # Optional: reset optimizer state
```

---

## Common Training Issues

### Issue 1: Out of Memory (OOM)

**Symptoms**: CUDA out of memory error during training

**Solutions**:
```bash
# Option 1: Reduce batch size
# Edit configs/training_config.yaml:
batch_size: 8  # From 16

# Option 2: Enable gradient checkpointing
use_gradient_checkpointing: true

# Option 3: Reduce num_workers
num_workers: 4  # From 8

# Option 4: Use smaller validation batch
val_batch_size: 16  # From 32
```

### Issue 2: Loss Not Decreasing

**Symptoms**: Loss stays constant or increases after warmup

**Checks**:
```bash
# 1. Verify data loading
python scripts/verify_data.py --tier 1

# 2. Check Zipformer checkpoint loaded
grep "Loaded Zipformer" outputs/tier1_run1/logs/training.log

# 3. Verify frozen encoder
# In logs, should see: "Zipformer encoder frozen: X parameters"

# 4. Reduce learning rate
learning_rate: 5e-5  # From 1e-4
```

### Issue 3: High WER (>20%)

**Symptoms**: Validation WER stuck above 20%

**Solutions**:
```bash
# 1. Increase ASR loss weight
asr_weight: 1.5  # From 1.0

# 2. Train longer
num_epochs: 75  # From 50

# 3. Reduce mel weight initially
mel_weight: 0.2  # From 0.5

# 4. Check data quality
python scripts/analyze_data.py --tier 1
```

### Issue 4: Low Consensus Score (<0.6)

**Symptoms**: Voting branches don't agree

**Solutions**:
```bash
# 1. Increase consensus weight
consensus_weight: 0.2  # From 0.1

# 2. Reduce codebook size
codebook_size: 2048  # From 4096 (in model_config.yaml)

# 3. Increase commitment weight
commitment_weight: 0.5  # From 0.25
```

---

## Training Best Practices

### Resource Management

**GPU Memory**:
- Monitor: `watch -n 1 nvidia-smi`
- Keep usage: <90% to allow headroom
- If OOM: Reduce batch_size before other changes

**Disk Space**:
- Monitor: `df -h`
- Checkpoints: ~500MB each, keep last 3
- Logs: ~10MB per epoch
- Clean old runs: `rm -rf outputs/old_run`

**Network Bandwidth** (for multi-GPU):
- Use fast interconnect (NVLink, InfiniBand)
- Monitor: `iftop -i eth0`

### Hyperparameter Tuning

**If WER too high**:
1. Increase `asr_weight`
2. Train longer
3. Reduce `mel_weight` initially
4. Check data quality

**If training unstable**:
1. Reduce `learning_rate`
2. Increase `warmup_epochs`
3. Enable `grad_clip`

**If consensus low**:
1. Increase `consensus_weight`
2. Reduce `codebook_size`
3. Increase `commitment_weight`

### Checkpointing Strategy

```bash
# Keep checkpoints at:
# - Every 5 epochs: For debugging
# - Best WER: For production
# - Latest: For resuming

# Clean old checkpoints (keep best + latest + last 2)
python scripts/clean_checkpoints.py --keep-n 4
```

---

## After Training Completes

### Step 1: Evaluate Best Model

```bash
# Evaluate on test set
python scripts/evaluate.py \
  --checkpoint outputs/tier1_run1/checkpoints/best_model.pt \
  --test-set librispeech-test-clean

# Expected output:
# Test WER: 8.5%
# Consensus Score: 0.83
# Mel Reconstruction L1: 0.42
```

### Step 2: Export to ONNX

```bash
# Export for iOS deployment
python scripts/export_onnx.py \
  --checkpoint outputs/tier1_run1/checkpoints/best_model.pt \
  --output-dir outputs/onnx_models

# Creates:
# outputs/onnx_models/
#   ├── voting_lfq.onnx        (~5MB, INT8 quantized)
#   ├── embedding.onnx         (~1MB, INT8 quantized)
#   └── adapter.onnx           (~0.5MB, INT8 quantized)
```

### Step 3: Validate ONNX Models

```bash
# Verify ONNX models produce same output as PyTorch
python scripts/validate_onnx.py \
  --pytorch-checkpoint outputs/tier1_run1/checkpoints/best_model.pt \
  --onnx-dir outputs/onnx_models

# Expected output:
# ✅ Voting-LFQ output matches (MSE: 1.2e-7)
# ✅ Embedding output matches (MSE: 3.4e-8)
# ✅ Adapter output matches (MSE: 5.6e-8)
# ✅ End-to-end tokens identical
```

### Step 4: Transfer to Dev Machine

```bash
# Compress ONNX models
tar -czf onnx_models_tier1.tar.gz outputs/onnx_models/

# Copy to dev machine
scp onnx_models_tier1.tar.gz user@dev-machine:~/ios-project/

# On dev machine:
tar -xzf onnx_models_tier1.tar.gz
# Extract to: SherpaOnnx/SherpaOnnx/Models/stable_token/
```

---

## Training Timeline Estimates

### Tier 1 (100 hours, LibriSpeech clean-100)

| Component | Time |
|-----------|------|
| Data loading & preprocessing | 2-3 hours |
| Training (50 epochs, 2× GPU) | 24-36 hours |
| Validation | 2-4 hours |
| ONNX export | 10 minutes |
| **Total** | **30-44 hours** |

### Tier 2 (1460 hours, Full dataset)

| Component | Time |
|-----------|------|
| Data loading & preprocessing | 6-8 hours |
| Training (100 epochs, 4× GPU) | 72-96 hours |
| Validation | 6-10 hours |
| ONNX export | 10 minutes |
| **Total** | **84-114 hours (3.5-4.7 days)** |

---

## Next Steps

1. **Monitor training**: TensorBoard + logs
2. **Wait for completion**: 1-5 days depending on tier
3. **Evaluate**: Test WER and metrics
4. **Export**: ONNX models for iOS
5. **Integrate**: See iOS integration guide

For troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
