<p align="center">
  <h1 align="center">🌊 SlipStream</h1>
  <p align="center"><strong>Run LLMs larger than your phone's RAM — layer by layer.</strong></p>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#supported-models">Models</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#building">Building</a>
</p>

---

## What is SlipStream?

SlipStream is a **pure C inference engine** for iOS and Android that uses **layer-streaming** to run LLMs far larger than device RAM. Instead of loading an entire model into memory, it streams transformer layers one at a time — **load, compute, evict** — keeping peak memory usage to just a single layer.

### Key Features

- 🧠 **Layer Streaming** — Only 1 layer in memory at a time. Run 7B models on 3GB RAM phones.
- ⚡ **Metal GPU** — iOS Metal compute shaders for fast matrix multiplication.
- 🔧 **ARM NEON** — SIMD-optimized CPU fallback for all ARM devices.
- 📦 **GGUF Format** — Compatible with thousands of pre-quantized models on HuggingFace.
- 🔒 **100% Offline** — No network calls. All inference happens on-device.
- 📱 **Native Bindings** — Swift (iOS) and Kotlin (Android) wrappers included.

## How It Works

Traditional inference loads ALL model weights into memory:

```
Traditional: [Layer 0][Layer 1][Layer 2]...[Layer 31] → 4-14GB RAM needed
```

SlipStream streams one layer at a time:

```
SlipStream:  [Layer 0] → compute → evict
             [Layer 1] → compute → evict
             [Layer 2] → compute → evict
             ...
             Peak RAM: ~150-400MB (1 layer + KV cache)
```

```
┌─────────────────────────────────┐
│          GGUF File (Disk)        │
│  ┌───┐┌───┐┌───┐┌───┐   ┌───┐ │
│  │ L0 ││ L1 ││ L2 ││ L3 │...│L31│ │
│  └─┬─┘└───┘└───┘└───┘   └───┘ │
└────┼────────────────────────────┘
     │ mmap
┌────▼────────────────────────────┐
│          Device RAM              │
│  ┌──────────┐ ┌──────────────┐  │
│  │ 1 Layer   │ │ KV Cache +   │  │
│  │ Weights   │ │ Activations  │  │
│  │ (~100MB)  │ │ (~50-300MB)  │  │
│  └──────────┘ └──────────────┘  │
│                                  │
│  Peak: 150-400MB vs 4-14GB      │
└──────────────────────────────────┘
```

## Supported Models

Any GGUF-format model works. Recommended models for mobile:

| Model | Parameters | GGUF Size (Q4) | Peak RAM | Quality |
|-------|-----------|-----------------|----------|---------|
| TinyLlama 1.1B | 1.1B | ~700MB | ~150MB | Basic |
| Phi-2 | 2.7B | ~1.7GB | ~200MB | Good |
| Llama 3.2 3B | 3B | ~1.9GB | ~250MB | Great |
| Mistral 7B | 7B | ~4.1GB | ~400MB | Excellent |
| Llama 3.1 8B | 8B | ~4.7GB | ~450MB | Excellent |

> Models need enough **disk space** (not RAM). A 64GB phone can store multiple 4GB models.

## Quickstart

### C API

```c
#include "slipstream.h"

// Load model
ss_model_t *model = ss_model_load("path/to/model.gguf", NULL);

// Generate
ss_generate_params_t params = ss_default_params();
params.max_tokens = 256;
params.temperature = 0.7f;

ss_generate(model, "Tell me about layer streaming:", &params,
    token_callback, NULL);

// Cleanup
ss_model_free(model);
```

### Swift (iOS)

```swift
import SlipStream

let model = try SlipStreamModel(path: "model.gguf")

for try await token in model.generate(prompt: "Hello!") {
    print(token, terminator: "")
}
```

## Building

### Requirements

- CMake 3.20+
- Xcode 15+ (for iOS/Metal)
- Android NDK r25+ (for Android)

### macOS / iOS

```bash
git clone https://github.com/user/slipstream.git
cd slipstream
mkdir build && cd build
cmake .. -DSLIPSTREAM_METAL=ON
make -j$(sysctl -n hw.ncpu)
```

### Tests

```bash
cmake .. -DBUILD_TESTS=ON
make -j$(sysctl -n hw.ncpu)
ctest --output-on-failure
```

## Benchmarks

*Coming soon — will be populated after testing on real devices.*

| Device | Model | Tokens/sec | Peak RAM | Backend |
|--------|-------|-----------|----------|---------|
| iPhone 15 Pro | Llama 3.2 3B Q4 | TBD | TBD | Metal |
| iPhone 13 | Phi-2 Q4 | TBD | TBD | Metal |
| Pixel 8 | Mistral 7B Q4 | TBD | TBD | CPU |

## Architecture

```
slipstream/
├── include/slipstream.h       # Public C API
├── src/
│   ├── core/                   # Engine, GGUF loader, layer scheduler
│   ├── backends/               # Metal (iOS), NEON (CPU)
│   └── utils/                  # Memory management, threading
├── bindings/
│   ├── swift/                  # iOS Swift wrapper
│   └── android/                # Android JNI + Kotlin
├── examples/                   # Demo apps
└── tests/                      # Unit + integration tests
```

## Comparison

| Feature | SlipStream | llama.cpp | MLC LLM |
|---------|-----------|-----------|---------|
| Layer streaming | ✅ Core feature | ❌ Full load | ❌ Full load |
| Peak RAM (7B Q4) | ~400MB | ~4.5GB | ~4.5GB |
| Max model size | Disk-limited | RAM-limited | RAM-limited |
| iOS Metal | ✅ | ✅ | ✅ |
| Android | ✅ | ✅ | ✅ |
| Speed | Slower (I/O bound) | Fastest | Fast |
| Offline | ✅ | ✅ | ✅ |

> **Tradeoff:** SlipStream is slower due to disk I/O per layer, but can run models that physically cannot fit in RAM on other frameworks.

## License

MIT — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Credits

Inspired by [AirLLM](https://github.com/lyogavin/airllm) by lyogavin and [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov.
