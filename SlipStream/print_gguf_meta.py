import sys, struct
from pathlib import Path

path = sys.argv[1]
with open(path, 'rb') as f:
    magic = f.read(4)
    if magic != b'GGUF':
        print("Not GGUF")
        sys.exit(1)
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    
    print(f"Version: {version}, Tensors: {n_tensors}, KV headers: {n_kv}")
    
    def read_str():
        strlen = struct.unpack('<Q', f.read(8))[0]
        return f.read(strlen).decode('utf-8', errors='ignore')
        
    for i in range(n_kv):
        key = read_str()
        type_id = struct.unpack('<I', f.read(4))[0]
        val = None
        if type_id == 2: # UINT32
            val = struct.unpack('<I', f.read(4))[0]
        elif type_id == 4: # UINT64
            val = struct.unpack('<Q', f.read(8))[0]
        elif type_id == 7: # FLOAT32
            val = struct.unpack('<f', f.read(4))[0]
        elif type_id == 8: # STRING
            val = read_str()
        else:
            # skip varying sizes for other types
            print(f"[{key}] uses unsupported type {type_id}")
            break
        print(f"{key}: {val}")
