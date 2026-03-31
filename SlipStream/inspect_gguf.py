import struct
import sys

def parse_gguf(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF": return
        version = struct.unpack("<I", f.read(4))[0]
        tensors = struct.unpack("<Q", f.read(8))[0]
        metadata_kv = struct.unpack("<Q", f.read(8))[0]
        
        print(f"GGUF Tensors: {tensors}, Metadata: {metadata_kv}")

        for _ in range(metadata_kv):
            l = struct.unpack("<Q", f.read(8))[0]
            key = f.read(l).decode('utf-8', errors='ignore')
            t = struct.unpack("<I", f.read(4))[0]
            
            if key in ["tokenizer.ggml.tokens", "tokenizer.ggml.scores"]:
                arr_type = struct.unpack("<I", f.read(4))[0]
                arr_len = struct.unpack("<Q", f.read(8))[0]
                print(f"Metadata Array [{key}]: type {arr_type}, length {arr_len}")
                # Don't try to parse rest, just exit since we found it
            elif t in (0,1,2,3,12): f.read(1)
            elif t in (4,5,10): f.read(2)
            elif t in (6,7,11): f.read(4)
            elif t in (8,9): f.read(8)
            elif t == 8: # string
                ls = struct.unpack("<Q", f.read(8))[0]
                f.read(ls)
            # The skip logic for the rest of metadata is too hard in python without full types.
            # Forget the rest of metadata, ill use grep on strings if it fails
            
parse_gguf("/Users/harrisonfarrell/Library/Developer/CoreSimulator/Devices/F63131EA-7456-41B8-B5F6-103559E63876/data/Containers/Shared/AppGroup/E61A30A0-4B7C-450C-9D2B-291B0CB92F26/File Provider Storage/Qwen3.5-4B-Q4_K_M.gguf")

