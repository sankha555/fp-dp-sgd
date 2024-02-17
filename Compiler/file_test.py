import struct, time
import numpy as np

file_path = "./Persistence/Transactions-P0.data"

# with open(file_path, 'r') as f:
#     line = f.read()
    
# print(line)

preamble = "\tZ2^64@"
mod = 2**64
n_bytes = 8
buf = []

def pack(val, buf):
    v = int(round(val)) % mod
    temp_buf = []
    for i in range(n_bytes):
        temp_buf.append(v & 0xff)
        v >>= 8
    #Instead of using python a loop per value we let struct pack handle all it
    buf += struct.pack('<{}B'.format(len(temp_buf)), *tuple(temp_buf))

def write_to_transaction(arr, party):
    file_path = f"./Persistence/Transactions-P{party}.data"
    
    buf = []
    pack(ord(preamble[0]), buf)
    for c in preamble[1:]:     
        buf.append(ord(c))
    buf.append(0)
    buf.append(0)
    buf.append(0)

    for eta in arr:
        pack(eta*(1<<16), buf)

    with open(file_path, "wb") as f:
        f.write(bytes(buf))

def sample_and_write(n):
    preamble = "\tZ2^64@"
    mod = 2**64
    n_bytes = 8
    buf = []
        
    tm = time.time()

    pack(ord(preamble[0]), buf)
    for c in preamble[1:]:     
        buf.append(ord(c))
    buf.append(0)
    buf.append(0)
    buf.append(0)

    noise = np.random.normal(0, 16, n)
    for eta in noise:
        pack(eta*(1<<16), buf)

    with open(file_path, "wb") as f:
        f.write(bytes(buf))

    # print(noise[:10])
    # print(time.time() - tm)
    
if __name__ == '__main__':
    sample_and_write(100)