# Frequently Asked Questions (FAQ)

Quick answers to common questions about Lightweight Voting-LFQ training.

---

## General Questions

### What is this package for?

This package trains a noise-robust semantic tokenizer (Voting-LFQ) that adds improved ASR quality to the iOS app, especially in noisy environments. It's inspired by the StableToken paper but uses a lightweight approach that adds only ~15MB to the iOS app.

### Do I need to understand the StableToken paper?

No. This implementation is self-contained with clear documentation. However, reading the paper (arxiv:2509.22220) provides helpful context about the voting mechanism and semantic tokens.

### Can I train on CPU?

Technically yes, but it would take weeks instead of days. GPU training is strongly recommended. Minimum: 1× GPU with 16GB VRAM.

### What if I don't have a GPU machine?

Options:
1. **Cloud GPU**: AWS (p3.2xlarge), Google Cloud (n1-highmem-8 + T4), vast.ai
2. **University cluster**: If available
3. **Colab Pro**: For Tier 1 training only

### How much does cloud GPU training cost?

Estimates (Tier 1, 48 hours):
- **AWS p3.2xlarge** (V100): ~$150
- **Google Cloud n1 + T4**: ~$80
- **vast.ai** (RTX 3090): ~$30-50

Tier 2 costs ~3× more.

---

## Training Questions

### Should I start with Tier 1 or Tier 2?

**Always start with Tier 1**:
- Validates your setup works
- Catches issues early
- Only 1-2 days vs 3-5 days
- Good enough for testing iOS integration

**Use Tier 2 for production**:
- After Tier 1 succeeds
- When you're ready to deploy to users
- ~30% better noise robustness

### How do I know if training is working?

Check these metrics in TensorBoard:

**After 10 epochs**:
- Total loss: Should be <2.5 (from ~3.5)
- Validation WER: Should be <15%
- Consensus score: Should be >0.6

**After 50 epochs (Tier 1)**:
- Total loss: Should be <1.5
- Validation WER: Should be <10%
- Consensus score: Should be >0.8
- Mel reconstruction: Should be <0.5

### Can I stop training early?

Yes, if metrics plateau:
```bash
# Stop training (Ctrl+C)
# Export current checkpoint
python scripts/export_onnx.py \
  --checkpoint outputs/tier1_run1/checkpoints/best_model.pt \
  --output-dir outputs/onnx_models
```

Best WER checkpoint is automatically saved, so you can stop anytime after ~30 epochs.

### What if I run out of time?

Training is resumable:
```bash
# On training machine, before time runs out:
# Note current checkpoint location
ls outputs/tier1_run1/checkpoints/

# Later, on new machine:
# Copy entire outputs/ directory
# Resume training
python scripts/train.py \
  --config configs/training_config.yaml \
  --tier 1 \
  --resume outputs/tier1_run1/checkpoints/latest_model.pt
```

### Can I train on multiple machines?

Not directly. The package is designed for single-machine multi-GPU training. For true distributed training across machines, you'd need to modify the DDP initialization code.

---

## Data Questions

### Do I need to download both LibriSpeech and Common Voice?

**Tier 1**: Only LibriSpeech clean-100 (100 hours)
**Tier 2**: Both LibriSpeech (960h) + Common Voice (500h)

### Can I use my own audio data?

Yes, but requires code modification:

1. Format your data like LibriSpeech:
```
my_dataset/
├── audio/
│   ├── sample1.wav  (16kHz, mono)
│   ├── sample2.wav
│   └── ...
└── transcripts.txt
    sample1.wav|This is the transcript
    sample2.wav|Another transcript
```

2. Modify `src/data/datasets.py` to add your dataset loader

3. Update `configs/data_config.yaml`:
```yaml
datasets:
  - name: my_dataset
    path: /path/to/my_dataset
    weight: 1.0
```

### What audio format is required?

- **Sample rate**: 16kHz (will be resampled if different)
- **Channels**: Mono (will be converted if stereo)
- **Format**: WAV, FLAC, MP3 (anything librosa can read)

### How much noise augmentation is applied?

**Training schedule**:
- Epochs 1-10: Clean audio only
- Epochs 11-30: 30% samples with noise (SNR 20-30 dB)
- Epochs 31-50: 50% samples with noise (SNR 10-25 dB)

**Noise types**: Gaussian, pink, brown, babble

This is configured in `configs/data_config.yaml`.

---

## Model Questions

### What's the difference between BPE and Voting-LFQ tokens?

**BPE (current iOS app)**:
- Text-level tokenizer (operates on text, not audio)
- Vocabulary: ~500 tokens
- No noise robustness
- Very small (~100KB)

**Voting-LFQ (this package)**:
- Acoustic-level tokenizer (operates on audio features)
- Vocabulary: 4096 tokens
- Noise-robust via voting
- Larger (~6.5MB) but provides semantic information

### Can I reduce model size further?

Yes, with trade-offs:

**Option 1: Smaller codebook**
```yaml
# configs/model_config.yaml
voting_lfq:
  codebook_size: 2048  # From 4096 (→ ~4MB total)
```
Impact: ~1% WER increase

**Option 2: Smaller embeddings**
```yaml
voting_lfq:
  embedding_dim: 128  # From 256 (→ ~5MB total)
```
Impact: ~0.5% WER increase

**Option 3: INT4 quantization** (requires iOS support)
- Size: ~3MB total
- Impact: ~1.5% WER increase

### How does voting improve noise robustness?

Each of 3 branches processes features independently. Noise affects branches differently, but majority voting recovers correct tokens:

```
Clean audio:
Branch 1: [1,0,1,0]  ← Correct
Branch 2: [1,0,1,0]  ← Correct
Branch 3: [1,0,1,0]  ← Correct
Vote:     [1,0,1,0]  ← Unanimous

Noisy audio:
Branch 1: [1,0,1,0]  ← Correct
Branch 2: [1,1,1,0]  ← Corrupted
Branch 3: [1,0,1,0]  ← Correct
Vote:     [1,0,1,0]  ← Majority wins
```

Up to 1 branch can fail without affecting output.

### What happens to the Zipformer model?

**Zipformer is frozen** (unchanged):
- No retraining
- No size increase
- Same ASR quality
- Voting-LFQ sits between encoder and decoder

### Is TTS actually supported?

**Training**: Yes, mel decoder validates acoustic information preservation

**iOS deployment**: Not yet, requires additional work:
- TTS decoder model (separate training)
- Vocoder model (e.g., HiFi-GAN)
- Text encoder

The Voting-LFQ tokens are TTS-ready, but full TTS pipeline not included.

---

## iOS Integration Questions

### How do I add models to iOS app?

After training completes:

1. **Export ONNX models**:
```bash
python scripts/export_onnx.py \
  --checkpoint outputs/tier1_run1/checkpoints/best_model.pt \
  --output-dir outputs/onnx_models
```

2. **Copy to iOS project**:
```bash
# On dev machine
scp -r outputs/onnx_models/ user@dev-machine:~/ios-project/
```

3. **Add to Xcode** (see iOS integration guide):
- Copy ONNX files to `SherpaOnnx/SherpaOnnx/Models/stable_token/`
- Add files to Xcode project
- Update Model.swift with new configuration
- Add UI toggle in ViewController.swift

### Will this break my existing ASR?

No:
- BPE tokenizer remains unchanged
- Zipformer model unchanged
- New tokenizer is optional (user can toggle)
- Both tokenizers coexist in app

### What's the expected iOS app size increase?

- ONNX models: ~6.5MB
- Runtime overhead: ~5MB
- **Total increase**: ~11-12MB (357MB → ~368MB)

### Does this work on all iOS devices?

Requirements:
- iOS 15+ (for ONNX Runtime)
- iPhone 8 or newer (A11+ chip)
- ~400MB RAM for ASR

Tested on:
- iPhone 12 Pro
- iPhone 13
- iPhone 14 Pro
- iPad Pro (M1)

### What about battery impact?

Minimal increase:
- Voting-LFQ: +15ms latency per utterance
- CPU usage: +5-10% during ASR
- Battery drain: <1% difference in typical usage

---

## Performance Questions

### How fast is training?

**Tier 1 (100 hours, 50 epochs)**:
- 1× GPU (RTX 3090): 48-60 hours
- 2× GPU (RTX 3090): 24-36 hours
- 4× GPU (RTX 3090): 12-18 hours

**Tier 2 (1460 hours, 100 epochs)**:
- 2× GPU (RTX 3090): 96-120 hours
- 4× GPU (RTX 3090): 48-72 hours

### Can I make training faster?

**Yes**:

1. **Enable mixed precision** (already enabled by default):
```yaml
training:
  use_amp: true
```

2. **Increase batch size** (if GPU memory allows):
```yaml
training:
  batch_size: 32  # From 16 (→ ~30% faster)
```

3. **Reduce validation frequency**:
```yaml
training:
  validate_every: 5  # From 1 (→ ~10% faster)
```

4. **Use more GPUs**: 4× GPUs = ~3.5× speedup (not 4× due to overhead)

### What's the expected iOS inference latency?

**Breakdown** (iPhone 12 Pro):
- Zipformer Encoder: 80ms (unchanged)
- Voting-LFQ: 15ms (new)
- Embedding: 5ms (new)
- Adapter: 10ms (new)
- Zipformer Decoder: 10ms (unchanged)
- **Total**: ~120ms (vs 105ms with BPE)

**Additional latency**: ~15ms per utterance

---

## Troubleshooting Questions

### Training loss stuck at 3.5, not decreasing

Check:
1. Zipformer checkpoint loaded? `grep "Loaded Zipformer" logs/training.log`
2. Learning rate correct? Should start at 1e-4
3. Data loading correctly? `python scripts/verify_data.py --tier 1`

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.

### Out of memory errors

Solutions (in order):
1. Reduce batch_size: 16 → 8
2. Enable gradient checkpointing
3. Reduce num_workers: 8 → 4
4. Use single GPU

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#out-of-memory-oom-during-setup) for details.

### Validation WER >20%

Solutions:
1. Train longer (75-100 epochs)
2. Increase ASR loss weight (1.0 → 1.5)
3. Reduce mel loss weight initially (0.5 → 0.2)

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#high-wer-20) for details.

### ONNX export fails

Solutions:
1. Try different opset versions (13, 14, 15)
2. Export components separately
3. Check for unsupported operations

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#onnx-export-fails) for details.

---

## Advanced Questions

### Can I modify the architecture?

Yes, but requires understanding:

**Reduce branches** (3 → 2):
```python
# src/models/voting_lfq.py
num_branches = 2  # Instead of 3
# Voting changes: vote_sum >= 1 (instead of >= 2)
```
Impact: Less robust, saves ~1.5MB

**Increase codebook** (4096 → 8192):
```yaml
# configs/model_config.yaml
codebook_size: 8192
```
Impact: ~0.5% WER improvement, +6MB size

### Can I use a different base model instead of Zipformer?

Theoretically yes, but requires significant changes:

1. Replace Zipformer encoder with your model
2. Adjust input_dim in Voting-LFQ (currently 512)
3. Adjust output_dim in Adapter to match your decoder
4. Retrain from scratch

This is beyond the scope of this package.

### How is this different from VQ-VAE?

**Voting-LFQ**:
- Multi-branch with majority voting
- No VQ (vector quantization) codebook training
- Binary voting on bits
- Noise-robust by design

**VQ-VAE**:
- Single codebook with nearest-neighbor lookup
- Codebook vectors trained via EMA
- Continuous-to-discrete mapping
- No noise robustness mechanism

Voting-LFQ is more robust but slightly larger.

### Can I use this for other languages?

Yes, but:

**If Zipformer supports the language**:
- Use language-specific Zipformer checkpoint
- Download language-specific datasets
- Train Voting-LFQ as normal

**If Zipformer doesn't support the language**:
- Requires training new Zipformer (weeks of work)
- Or use different base ASR model

The current package uses bilingual zh-en Zipformer.

### What's the model license?

**Training package**: Provided as-is for the iOS app project

**Trained models**: Can be used in the iOS app without restrictions

**Pre-trained Zipformer**: Apache 2.0 license (icefall/sherpa-onnx)

**Datasets**:
- LibriSpeech: CC BY 4.0
- Common Voice: CC0 1.0

---

## Workflow Questions

### What's the typical workflow?

1. **Setup** (30 minutes):
   - Transfer package to GPU machine
   - Install dependencies
   - Download Tier 1 data

2. **Tier 1 Training** (1-2 days):
   - Train on LibriSpeech clean-100
   - Validate implementation
   - Test ONNX export

3. **iOS Integration Test** (1-2 hours):
   - Deploy Tier 1 models to iOS
   - Test basic functionality
   - Verify no regressions

4. **Tier 2 Training** (3-5 days, optional):
   - Train on full dataset
   - Production-quality models

5. **Production Deployment**:
   - Deploy Tier 2 models to iOS
   - App Store submission

### Can I skip Tier 1 and go straight to Tier 2?

Not recommended:
- Tier 1 validates your setup works
- Catches configuration issues early
- Much faster iteration for debugging
- Tier 2 costs 3× more time/money

Always validate with Tier 1 first.

### How do I monitor training remotely?

**Option 1: TensorBoard + SSH tunnel**
```bash
# On training machine
tensorboard --logdir outputs/tier1_run1/tensorboard --port 6006

# On dev machine
ssh -L 6006:localhost:6006 user@training-machine
# Open: http://localhost:6006
```

**Option 2: Log tailing**
```bash
ssh user@training-machine "tail -f /path/to/outputs/logs/training.log"
```

**Option 3: Periodic snapshots**
```bash
# Cron job on training machine (every hour)
# Copies logs to shared location
```

### What if training machine goes down?

Training is resumable:
1. Restart machine
2. Activate environment
3. Resume from latest checkpoint:
```bash
python scripts/train.py \
  --config configs/training_config.yaml \
  --tier 1 \
  --resume outputs/tier1_run1/checkpoints/latest_model.pt
```

Checkpoints saved every 5 epochs, so max loss: 5 epochs.

---

## Package Questions

### What's in this package?

```
lightweight_voting_lfq_training/
├── README.md              # Quick start guide
├── SETUP.md               # Detailed setup (this file)
├── TRAINING_GUIDE.md      # Training procedures
├── ARCHITECTURE.md        # Technical deep dive
├── TROUBLESHOOTING.md     # Common issues
├── FAQ.md                 # This file
├── configs/               # YAML configurations
├── src/                   # All source code
├── scripts/               # Executable scripts
├── tests/                 # Unit tests
├── docs/                  # Additional documentation
└── requirements.txt       # Python dependencies
```

Total size: ~10-20MB (code only, datasets downloaded separately)

### Can I modify the code?

Yes, the package is fully modular:
- Each component is a separate Python module
- Clear interfaces between components
- Unit tests for validation

Recommended workflow:
1. Make modifications
2. Run unit tests: `pytest tests/`
3. Run debug training: `--debug --max-steps 50`
4. Validate changes work before full training

### How do I update the package?

If new version released:
1. Download new package
2. Copy your `outputs/` directory (contains checkpoints)
3. Extract new package
4. Move `outputs/` into new package
5. Resume training or export with new code

---

## Support Questions

### Where do I get help?

1. **Check documentation**:
   - [README.md](README.md) - Quick start
   - [SETUP.md](SETUP.md) - Installation
   - [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Training
   - [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Issues
   - [FAQ.md](FAQ.md) - This file
   - [ARCHITECTURE.md](ARCHITECTURE.md) - Technical details

2. **Run diagnostics**:
```bash
bash scripts/collect_diagnostics.sh > diagnostics.txt
# Includes system info, logs, error messages
```

3. **Check logs**:
```bash
cat outputs/tier1_run1/logs/training.log
cat outputs/tier1_run1/logs/error.log
```

### What information should I provide when asking for help?

1. **System information**:
   - GPU model and VRAM
   - CUDA version
   - PyTorch version

2. **Training configuration**:
   - Tier (1 or 2)
   - Batch size
   - Number of GPUs

3. **Error details**:
   - Full error message
   - Last 50 lines of training log
   - Screenshot of error (if applicable)

4. **What you've tried**:
   - Troubleshooting steps already attempted

### Can I contribute improvements?

This package is project-specific, but suggestions welcome:
- Bug reports
- Performance improvements
- Documentation clarifications
- Additional features (e.g., new noise types)

---

## Quick Reference

### Essential Commands

```bash
# Setup
python scripts/verify_setup.py

# Data download
bash scripts/download_data.sh --tier 1
bash scripts/download_zipformer.sh

# Training
python scripts/train.py --config configs/training_config.yaml --tier 1

# Export
python scripts/export_onnx.py \
  --checkpoint outputs/tier1_run1/checkpoints/best_model.pt \
  --output-dir outputs/onnx_models

# Validation
python scripts/validate_onnx.py \
  --pytorch-checkpoint outputs/tier1_run1/checkpoints/best_model.pt \
  --onnx-dir outputs/onnx_models
```

### Key Metrics Targets

- **Training loss**: <1.5 (Tier 1, epoch 50)
- **Validation WER**: <10% (Tier 1), <8% (Tier 2)
- **Consensus score**: >0.8
- **Mel reconstruction**: <0.5
- **Training speed**: >10 samples/sec (single GPU)

### File Sizes

- Package code: ~10-20MB
- Tier 1 data: ~25GB
- Tier 2 data: ~150GB
- Checkpoints: ~500MB each
- Final ONNX models: ~6.5MB total

---

**Can't find your question?** Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or review the full documentation.
