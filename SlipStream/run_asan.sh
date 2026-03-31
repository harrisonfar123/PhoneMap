#!/bin/bash
cd /Users/harrisonfarrell/Desktop/IB/embeddingphoine

# Clean previous build
rm -rf build_asan
mkdir build_asan
cd build_asan

# Build with ASAN
cmake -DCMAKE_C_FLAGS="-fsanitize=address -g -O0" -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" ..
make -j4 slipstream-cli

# Run CLI
./slipstream-cli /Users/harrisonfarrell/Library/Developer/CoreSimulator/Devices/F63131EA-7456-41B8-B5F6-103559E63876/data/Containers/Shared/AppGroup/E61A30A0-4B7C-450C-9D2B-291B0CB92F26/File\ Provider\ Storage/Qwen3.5-4B-Q4_K_M.gguf "The secret to happiness is" > asan_log.txt 2>&1

cat asan_log.txt
