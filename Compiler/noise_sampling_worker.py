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

try:
    std = int(sys.argv[1])
except:
    std = 16
    
try:
    n_parties = int(sys.argv[2])
except:
    n_parties = 1
    
try:
    n_samples = int(sys.argv[3])
except:
    raise Exception("Number of samples should be specified")

try:
    double = int(sys.argv[4])
except:
    double = 0
    
if double and n_parties > 1:
    raise Exception("Double noise can only be simulated for single party!")

try:
    numpy_double = int(sys.argv[5])
except:
    numpy_double = 0

if numpy_double and (double == 0):
    raise Exception("Numpy noise can be used only when double noise is being added by a single party!")

try:
    n_iterations = int(sys.argv[6])
except:
    raise Exception("Specify number of iterations!")
    
# party = int(sys.argv[2])
client_id = 0

client = Client(['localhost'] * n_parties, 15000, client_id)

class SamplingHandler:
    n_cores = 4
    proc_start_times = [0]*n_cores 
    proc_end_times = [0]*n_cores
    
    def __init__(self, total, party):
        print(f"Generating {total} samples for party {party}")
        division = math.ceil(total / SamplingHandler.n_cores)
        ns_per_process = [division]*SamplingHandler.n_cores        
        
        if total % SamplingHandler.n_cores > 0:
            ns_per_process[-1] = total % division
        
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
            
    def sampling_process(self, n, id):
        res = get_noise_vector(std, n)
        with open(f"./NoiseSamples/party{self.party}_p{id}.txt", "w+") as f:
            f.writelines(str(res))
            
    def parse_res(self):
        res = []
        for i in range(SamplingHandler.n_cores):
            with open(f"./NoiseSamples/party{self.party}_p{i}.txt", "r") as f:
                res_part = eval(f.readline())
                res.extend(res_part)    
            
        return res


def write_to_party(party):
    tm = time.time()
    os = octetStream()
    
    res = []
    if numpy_double:
        res1 = np.random.normal(0, std, n_samples)
        res2 = np.random.normal(0, std, n_samples)
        res = (res1 + res2).tolist()
    else:
        sp = SamplingHandler(n_samples, party)
        res = sp.parse_res()  
        
        if double:
            sp1 = SamplingHandler(n_samples, party)
            res1 = sp1.parse_res()
            assert len(res) == len(res1)
            res = [res[i] + res1[i] for i in range(len(res))]
    
    print(len(res))
    write_to_transaction(res, party)
    
    time.sleep(2)
    
    SIG = party+1
    secs = time.time() - tm 
    print(f"Party {party}: {len(res)} samples in {secs:.2f} s")
    
    client.domain(SIG).pack(os)
    os.Send(client.sockets[party])

i = 0
while i < n_iterations:        
    for party in range(n_parties):
        write_to_party(party)
    i += 1
