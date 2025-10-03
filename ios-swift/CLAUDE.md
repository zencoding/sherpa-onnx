# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an iOS Swift application demonstrating real-time speech recognition (ASR) using sherpa-onnx, a cross-platform speech recognition library. The app uses ONNX Runtime for neural network inference with streaming zipformer/transducer models.

**Repository Context**: This project is located in `ios-swift/` within the larger sherpa-onnx repository at `https://github.com/k2-fsa/sherpa-onnx`, which provides speech recognition APIs across multiple platforms and languages.

## Build System

The app uses Xcode for building and depends on the sherpa-onnx.xcframework which must be built separately.

### Building the Framework

Before opening the Xcode project, you must build the sherpa-onnx.xcframework:

```bash
# From the parent directory (sherpa-onnx root)
cd ..
./build-ios.sh
```

This script:
- Downloads ONNX Runtime 1.17.1 for iOS
- Builds sherpa-onnx for iOS simulator (x86_64) and device (arm64)
- Creates `build-ios/sherpa-onnx.xcframework`

### Building the iOS App

```bash
# Open the Xcode project
open SherpaOnnx/SherpaOnnx.xcodeproj

# Or build from command line
xcodebuild -project SherpaOnnx/SherpaOnnx.xcodeproj -scheme SherpaOnnx -sdk iphoneos
```

## Required Model Files

The app requires pre-trained ONNX models to be added to the Xcode project as bundle resources:

**Current Configuration** (bilingual Chinese/English):
- `encoder-epoch-99-avg-1.onnx` (~330 MB)
- `decoder-epoch-99-avg-1.onnx` (~14 MB)
- `joiner-epoch-99-avg-1.onnx` (~13 MB)
- `tokens.txt` (vocabulary file)
- `bpe.model` and `bpe.vocab` (tokenizer files)

**Location**: `SherpaOnnx/` directory (at project root level)

**Adding Models to Xcode**:
1. Download model from https://k2-fsa.github.io/sherpa/onnx/pretrained_models/
2. Add files to Xcode project
3. Ensure files are in "Build Phases → Copy Bundle Resources"

## Architecture

### Core Components

**ViewController.swift** - Main UI controller
- Manages audio recording via AVAudioEngine
- Implements real-time audio processing pipeline
- Handles streaming recognition with incremental result updates
- Uses audio tap on input node with format conversion (device format → 16kHz mono float32)

**Model.swift** - Model configuration
- Factory functions for different pre-trained models (zipformer, paraformer)
- Resource bundle path resolution
- Model selection: `getBilingualStreamZhEnZipformer20230220()` (current default)

**Swift API Bridge** - Located in `../swift-api-examples/`
- `SherpaOnnx.swift`: Swift wrapper around C API
- `SherpaOnnx-Bridging-Header.h`: Objective-C bridge to C API
- Provides Swift-friendly interface to sherpa-onnx C library

### Audio Processing Pipeline

```
AVAudioEngine (device format, 48kHz)
  ↓
Audio Tap (512 frame buffer)
  ↓
AVAudioConverter (→ 16kHz mono float32)
  ↓
SherpaOnnxRecognizer.acceptWaveform()
  ↓
recognizer.decode() (streaming)
  ↓
recognizer.getResult().text (incremental updates)
  ↓
UI Label Update (main thread)
```

### Recognition Flow

1. **Initialization**: Create recognizer with model config and feature config (16kHz, 80-dim features)
2. **Streaming**: Audio chunks continuously fed to recognizer
3. **Decoding**: `recognizer.isReady()` → `recognizer.decode()` loop
4. **Endpoint Detection**: `recognizer.isEndpoint()` detects sentence boundaries
5. **Result Updates**: Incremental text updates during speech, finalized on endpoint

## Project Dependencies

### Framework Dependencies
- **sherpa-onnx.xcframework** (custom, from parent repo build)
  - Path: `../../build-ios/sherpa-onnx.xcframework`
  - Contains C++ library and C API headers

- **ONNX Runtime** (bundled in sherpa-onnx.xcframework)
  - Version: 1.17.1
  - Provides neural network inference

### iOS Frameworks
- AVFoundation (audio recording and processing)
- UIKit (UI components)

### Linking Configuration
- Bridging header points to: `../../../swift-api-examples/SherpaOnnx-Bridging-Header.h`
- Swift API wrapper: `../../../swift-api-examples/SherpaOnnx.swift`

## Development Workflow

### Adding New Models

1. Download model from https://k2-fsa.github.io/sherpa/onnx/pretrained_models/
2. Add factory function in `Model.swift` following existing patterns
3. Update model selection in `ViewController.initRecognizer()`
4. Add model files to Xcode project bundle resources

### Modifying Recognition Parameters

Edit `ViewController.initRecognizer()`:
- `sampleRate`: 16000 (fixed for most models)
- `featureDim`: 80 (model-dependent)
- `enableEndpoint`: true (automatic sentence segmentation)
- `rule1MinTrailingSilence`: 2.4s (endpoint detection)
- `rule2MinTrailingSilence`: 0.8s (endpoint detection)
- `decodingMethod`: "greedy_search" or "modified_beam_search"

### Testing Changes

```bash
# Build and run on simulator
xcodebuild -project SherpaOnnx/SherpaOnnx.xcodeproj \
  -scheme SherpaOnnx \
  -sdk iphonesimulator \
  -destination 'platform=iOS Simulator,name=iPhone 15'

# Or use Xcode GUI: Cmd+R
```

## Common Issues

### Missing Framework Error
**Error**: "sherpa-onnx.xcframework not found"
**Solution**: Run `../build-ios.sh` from parent directory

### Missing Model Files Error
**Error**: "encoder-epoch-99-avg-1.onnx does not exist"
**Solution**: Add model files to "Build Phases → Copy Bundle Resources"

### Audio Format Conversion Issues
**Symptom**: Recognizer receives no/incorrect audio
**Check**: Input format conversion from device format to 16kHz mono float32 in `initRecorder()`

### Microphone Permission
**Issue**: App doesn't receive audio
**Check**: Info.plist contains "Privacy - Microphone Usage Description" key

## Code Conventions

- **Swift Naming**: camelCase for functions/variables, PascalCase for types
- **Audio Buffer Processing**: Extensions on `AudioBuffer` and `AVAudioPCMBuffer` for array conversion
- **UI Updates**: Always dispatch to main thread via `DispatchQueue.main.async`
- **Resource Loading**: Use `getResource()` helper with precondition checks
- **Model Configuration**: Factory functions return complete config structs

## Related Documentation

- Sherpa-ONNX main docs: https://k2-fsa.github.io/sherpa/onnx/
- Pre-trained models: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/
- Parent repository: https://github.com/k2-fsa/sherpa-onnx
- Swift API examples: `../swift-api-examples/`
- Build scripts: `../build-ios.sh`, `../build-ios-shared.sh`
