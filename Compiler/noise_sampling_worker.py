#!/usr/bin/python3

import sys, random
from ExternalIO.client import *
from Compiler.discretegauss import get_noise_vector, scaled_noise_sample
import numpy as np
import torch
from numba import njit, jit, prange
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import math
from Compiler.file_test import sample_and_write, write_to_transaction

def sampler(n):
    # print(f"Noise {n}")
    get_noise_vector(16, n)


try:
    std = sys.argv[1]
except:
    std = 16
    
try:
    n_parties = int(sys.argv[2])
except:
    n_parties = 1
    
# party = int(sys.argv[2])
client_id = 0

client = Client(['localhost'] * n_parties, 15000, client_id)
sampler = lambda n: get_noise_vector(16, n)

class SamplingHandler:
    n_cores = 4
    proc_start_times = [0]*n_cores 
    proc_end_times = [0]*n_cores
    
    def __init__(self, total, party):
        division = total // SamplingHandler.n_cores
        ns_per_process = [division]*SamplingHandler.n_cores if total % SamplingHandler.n_cores == 0 else [division]*(SamplingHandler.n_cores - 1) + [total % SamplingHandler.n_cores]
        self.party = party
                            
        procs = []
        for i in range(SamplingHandler.n_cores):
            this_n = ns_per_process[i]
            p = mp.Process(target=self.sampling_process, args=(this_n, i))
            procs.append(p)
            
        for i in range(SamplingHandler.n_cores):
            # print(i)
            # SamplingHandler.proc_start_times[i] = time.time()
            procs[i].start()
        
        for i in range(SamplingHandler.n_cores):
            procs[i].join()
            # SamplingHandler.proc_end_times[i] = time.time()
            # print(f"Process {i} = {SamplingHandler.proc_end_times[i] - SamplingHandler.proc_start_times[i]} s")
            
    def parse_res(self):
        res = []
        for i in range(SamplingHandler.n_cores):
            with open(f"./party{self.party}_p{i}.txt", "r") as f:
                res_part = eval(f.readline())
                res.extend(res_part)    
            
        return res
            
    def sampling_process(self, n, id):
        res = get_noise_vector(16, n)
        with open(f"./party{self.party}_p{id}.txt", "w") as f:
            f.writelines(str(res))

tm = time.time()

def write_to_party(party):
    os = octetStream()
    sp = SamplingHandler(400000, party)
    res = sp.parse_res()   
    write_to_transaction(res, party)
    
    time.sleep(2)
    
    SIG = party+1
    print(SIG)
    client.domain(SIG).pack(os)
    os.Send(client.sockets[party])

i = 0
while i < 1000:
    for party in range(n_parties):
        write_to_party(party)
    i += 1
