// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SlipStream",
    defaultLocalization: "en",
    platforms: [
        .macOS(.v13), .iOS(.v15)
    ],
    products: [
        .library(name: "SlipStream", targets: ["SlipStream"]),
    ],
    targets: [
        .target(
            name: "CSlipStream",
            path: ".",
            exclude: [
                "build", "models", "bindings", "examples", "tests", 
                "README.md", "CMakeLists.txt", "LICENSE",
                ".git", ".gitignore"
            ],
            sources: [
                "src/core/engine.c",
                "src/core/gguf_loader.c",
                "src/core/layer_scheduler.c",
                "src/core/quantize.c",
                "src/core/tensor.c",
                "src/core/tokenizer.c",
                "src/core/SlipStreamVocabValidator.cpp",
                "src/core/SlipStreamSafeBuffers.cpp",
                "src/core/SlipStreamSoftmax.cpp",
                "src/core/SlipStreamSampling.cpp",
                "src/backends/cpu_neon.c",
                "src/backends/metal_backend.m",
                "src/backends/SlipStreamMemoryPool.m",
                "src/backends/SlipStreamGGUFMapper.m",
                "src/backends/SlipStreamTensor.m",
                "src/utils/memory.c",
                "src/utils/thread_pool.c",
                "src/utils/profiler.c",
                "src/utils/device_info.m"
            ],
            resources: [
                .process("src/backends/slipstream_safe_kernels.metal")
            ],
            publicHeadersPath: "include",
            cSettings: [
                .define("SS_HAS_METAL"),
                .define("SS_HAS_NEON"),
                .headerSearchPath("src")
            ],
            cxxSettings: [
                .headerSearchPath("src"),
                .define("SS_HAS_METAL"),
                .define("SS_HAS_NEON")
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                .linkedFramework("Foundation")
            ]
        ),
        .target(
            name: "SlipStream",
            dependencies: ["CSlipStream"],
            path: "bindings/swift/Sources/SlipStream"
        ),
        .testTarget(
            name: "SlipStreamTests",
            dependencies: ["SlipStream"],
            path: "bindings/swift/Tests/SlipStreamTests"
        )
    ],
    cxxLanguageStandard: .cxx17
)
