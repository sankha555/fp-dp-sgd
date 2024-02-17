import math
import re

from Compiler import mpc_math, util
from Compiler.types import *
from Compiler.types import _unreduced_squant
from Compiler.library import *
from Compiler.util import is_zero, tree_reduce
from Compiler.comparison import CarryOutRawLE
from Compiler.GC.types import sbitint
from functools import reduce
from typing import List
import time as tm

from Compiler.discretegauss import sample_dgauss, scaled_noise_sample, get_noise_vector
import numpy as np
import math
import random #Default random number generator,
import requests
from fractions import Fraction #we will work with rational numbers
import signal

use_mux = False    

def log_e(x):
    return mpc_math.log_fx(x, math.e)

def set_n_threads(n_threads):
    Layer.n_threads = n_threads
    SGD.n_threads = n_threads

def set_batch_size(batch_size):
    Optimizer.batch_size = batch_size
    
def exp(x):
    if use_mux:
        return mpc_math.mux_exp(math.e, x)
    else:
        return mpc_math.pow_fx(math.e, x)

def get_limit(x):
    exp_limit = 2 ** (x.k - x.f - 1)
    return math.log(exp_limit)

def sanitize(x, raw, lower, upper):
    limit = get_limit(x)
    res = (x > limit).if_else(upper, raw)
    return (x < -limit).if_else(lower, res)

def sigmoid(x):
    """ Sigmoid function.

    :param x: sfix """
    return sigmoid_from_e_x(x, exp(-x))

def sigmoid_from_e_x(x, e_x):
    return sanitize(x, 1 / (1 + e_x), 0, 1)

def sigmoid_prime(x):
    """ Sigmoid derivative.

    :param x: sfix """
    sx = sigmoid(x)
    return sx * (1 - sx)

@vectorize
def approx_sigmoid(x, n=3):
    """ Piece-wise approximate sigmoid as in
    `Hong et al. <https://arxiv.org/abs/2002.04344>`_

    :param x: input
    :param n: number of pieces, 3 (default) or 5
    """
    if n == 5:
        cuts = [-5, -2.5, 2.5, 5]
        le = [0] + [x <= cut for cut in cuts] + [1]
        select = [le[i + 1] - le[i] for i in range(5)]
        outputs = [cfix(10 ** -4),
                   0.02776 * x + 0.145,
                   0.17 * x + 0.5,
                   0.02776 * x + 0.85498,
                   cfix(1 - 10 ** -4)]
        return sum(a * b for a, b in zip(select, outputs))
    else:
        a = x < -0.5
        b = x > 0.5
        return a.if_else(0, b.if_else(1, 0.5 + x))

def lse_0_from_e_x(x, e_x):
    return sanitize(-x, log_e(1 + e_x), x + 2 ** -x.f, 0)

def lse_0(x):
    return lse_0_from_e_x(x, exp(x))

def approx_lse_0(x, n=3):
    assert n != 5
    a = x < -0.5
    b = x > 0.5
    return a.if_else(0, b.if_else(x, 0.5 * (x + 0.5) ** 2)) - x

def relu_prime(x):
    """ ReLU derivative. """
    return (0 <= x)

def relu(x):
    """ ReLU function (maximum of input and zero). """
    return (0 < x).if_else(x, 0)

def argmax(x):
    """ Compute index of maximum element.

    :param x: iterable
    :returns: sint or 0 if :py:obj:`x` has length 1
    """
    def op(a, b):
        comp = (a[1] > b[1])
        return comp.if_else(a[0], b[0]), comp.if_else(a[1], b[1])
    return tree_reduce(op, enumerate(x))[0]

def softmax(x):
    """ Softmax.

    :param x: vector or list of sfix
    :returns: sfix vector
    """
    sftmax = softmax_from_exp(exp_for_softmax(x)[0])
    # print_ln("SOFTMAX = %s", sftmax.reveal())
    return sftmax

def exp_for_softmax(x):
    m = util.max(x) - get_limit(x[0]) + math.log(len(x))
    mv = m.expand_to_vector(len(x))
    try:
        x = x.get_vector()
    except AttributeError:
        x = sfix(x)
    if use_mux:
        return exp(x - mv), m
    else:
        return (x - mv > -get_limit(x)).if_else(exp(x - mv), 0), m

def softmax_from_exp(x):
    y = x / sum(x)
    # print_ln("sum_y = %s", sum(y).reveal())
    return y

report_progress = False

def progress(x):
    if report_progress:
        print_ln(x)
        time()
        
def get_L2_norm_sq(vec, sc=1):
    scale = sc
    vec = vec*(sfix(1/scale).expand_to_vector(len(vec)))
    sum_of_sq = sum(vec * vec)
    
    return sum_of_sq*scale*scale
   
def get_L2_norm(vec, sc=1):
    scale = sc
    vec = vec*(sfix(1/scale).expand_to_vector(len(vec)))
    sum_of_sq = sum(vec * vec)
    L2_norm = mpc_math.sqrt(sum_of_sq)*(scale)
    return L2_norm

def clip_vector(vec, clip):
    scale = 1
    L2_norm = get_L2_norm(vec, scale)
    mul = util.if_else(L2_norm > clip, (clip/L2_norm), sfix(1)).expand_to_vector(len(vec))
    
    vec = vec * mul
    return vec*(sfix(scale).expand_to_vector(len(vec)))

   
class Optimizer:
    iterations_done = 0
    clip = 0
    noisy = 0 
    noise_bag_cursor = 0
    double_noise = False
    batch_size = 1
    
    def get_noise(self, sigma, n):
        self.noise_store.input_from(0, n)
        return self.noise_store

    def get_noise_np(self, sigma, n):        
        @for_range_opt(10)
        def _(i):
            self.noise_bag[i] = scaled_noise_sample(sigma)
            
        print_ln("NOISE = %s", self.noise_bag.get_vector(0, 10).reveal())
    
    def get_noise_from_file(self, n):
        noise = sfix.Array(n)
        @for_range_opt(n)
        def _(i):
            unif_ind = regint.get_random(17)
            noise[i] = self.noise_bag[unif_ind]
            if self.double_noise:
                unif_ind = regint.get_random(17)
                noise[i] += self.noise_bag[unif_ind]
                
        # print_ln("NOISE = %s", noise.get_vector(0, 10).reveal())
        return noise.get_vector()   
    
    def get_noise_samples_from_worker(self, std, n=3):        
        print_ln("Reading noise")
        @for_range(n)
        def _(i):
            @if_(get_player_id()._v == 0)
            def _():
                global x
                self.noise_bag[i] = sfix(personal(0, cfix.read_from_socket(socket)))
        
        print_ln("Read noise")
        self.noise_bag_cursor = 0
        
    def get_noise_samples_from_worker_file(self, n=3):
        # print_str("Waiting...")
        @for_range(1000)
        def _(i):
            @if_(get_player_id()._v == 0)
            def _():
                global sig
                sig =  regint.read_from_socket(socket)
                # print_ln("Player %s: SIG = %s", 0, sig.reveal())
            
                @if_(sig == 1)
                def _():
                    break_loop()
                                
            
            @if_(get_player_id()._v == 1)
            def _():
                global sig
                sig =  regint.read_from_socket(socket)
                # print_ln("Player %s: SIG = %s", 1, sig.reveal())
            
                @if_(sig == 2)
                def _():
                    break_loop()
                        
        self.noise_bag.read_from_file(0)
        # print_ln("NOISE = %s", self.noise_bag.get_vector(0, 10).reveal())

class Tensor(MultiArray):
    def __init__(self, *args, **kwargs):
        kwargs['alloc'] = False
        super(Tensor, self).__init__(*args, **kwargs)

    def input_from(self, *args, **kwargs):
        self.alloc()
        super(Tensor, self).input_from(*args, **kwargs)

    def __getitem__(self, *args):
        self.alloc()
        return super(Tensor, self).__getitem__(*args)

    def assign_all(self, *args):
        self.alloc()
        return super(Tensor, self).assign_all(*args)

    def assign_vector(self, *args, **kwargs):
        self.alloc()
        return super(Tensor, self).assign_vector(*args, **kwargs)

    def assign_vector_by_indices(self, *args):
        self.alloc()
        return super(Tensor, self).assign_vector_by_indices(*args)

class Layer:
    thetas = lambda self: (self.W, self.b)
    nablas = lambda self: (self.nabla_W, self.nabla_b)
    per_sample_norms = lambda self: (self.nabla_W_per_sample_norms, self.nabla_b_per_sample_norms)
    
    n_threads = 1
    inputs = []
    input_bias = True
    debug_output = False
    back_batch_size = 1e9
    print_random_update = False
    compute_nabla_X = False

    optimizer: Optimizer = None

    @property
    def shape(self):
        return list(self._Y.sizes)

    @property
    def X(self):
        self._X.alloc()
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    @property
    def Y(self):
        self._Y.alloc()
        return self._Y

    @Y.setter
    def Y(self, value):
        self._Y = value

    def forward(self, batch=None, training=None):
        if batch is None:
            batch = Array.create_from(regint(0))
        self._forward(batch)

    def __str__(self):
        return type(self).__name__ + str(self._Y.shape)

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, self.Y.shape)

class NoVariableLayer(Layer):
    input_from = lambda *args, **kwargs: None
    output_weights = lambda *args: None
    reveal_parameters_to_binary = lambda *args, **kwargs: None

    nablas = lambda self: ()
    reset = lambda self: None

class ElementWiseLayer(NoVariableLayer):
    def __init__(self, shape, inputs=None):
        self.X = Tensor(shape, sfix)
        self.Y = Tensor(shape, sfix)
        backward_shape = list(shape)
        backward_shape[0] = min(shape[0], self.back_batch_size)
        self.nabla_X = Tensor(backward_shape, sfix)
        self.nabla_Y = Tensor(backward_shape, sfix)
        self.inputs = inputs

    def f_part(self, base, size):
        return self.f(self.X.get_vector(base, size))

    def f_prime_part(self, base, size):
        return self.f_prime(self.Y.get_vector(base, size))

    def _forward(self, batch=[0]):
        n_per_item = reduce(operator.mul, self.X.sizes[1:])
        @multithread(self.n_threads, len(batch) * n_per_item)
        def _(base, size):
            self.Y.assign_vector(self.f_part(base, size), base)

        if self.debug_output:
            name = self
            @for_range(len(batch))
            def _(i):
                print_ln('%s X %s %s', name, i, self.X[i].reveal_nested())
                print_ln('%s Y %s %s', name, i, self.Y[i].reveal_nested())

    def _backward(self, batch):
        f_prime_bit = MultiArray(self.X.sizes, self.prime_type)
        n_elements = len(batch) * reduce(operator.mul, f_prime_bit.sizes[1:])

        @multithread(self.n_threads, n_elements)
        def _(base, size):
            f_prime_bit.assign_vector(self.f_prime_part(base, size), base)

        progress('f prime')

        @multithread(self.n_threads, n_elements)
        def _(base, size):
            self.nabla_X.assign_vector(self.nabla_Y.get_vector(base, size) *
                                       f_prime_bit.get_vector(base, size),
                                       base)

        # print_ln("RELU_NABLA_X = %s", self.nabla_X.get_vector(0, 100).reveal())
        
        progress('f prime schur Y')

        if self.debug_output:
            name = self
            @for_range(len(batch))
            def _(i):
                print_ln('%s X %s %s', name, i, self.X[i].reveal_nested())
                print_ln('%s f_prime %s %s', name, i, f_prime_bit[i].reveal_nested())
                print_ln('%s nabla Y %s %s', name, i, self.nabla_Y[i].reveal_nested())
                print_ln('%s nabla X %s %s', name, i, self.nabla_X[i].reveal_nested())

class Relu(ElementWiseLayer):
    """ Fixed-point ReLU layer.

    :param shape: input/output shape (tuple/list of int)
    """
    prime_type = sint

    def __init__(self, shape, inputs=None):
        super(Relu, self).__init__(shape)
        self.comparisons = MultiArray(shape, sint)

    def f_part(self, base, size):
        x = self.X.get_vector(base, size)
        c = x > 0
        self.comparisons.assign_vector(c, base)
        return c.if_else(x, 0)

    def f_prime_part(self, base, size):
        return self.comparisons.get_vector(base, size)
    
class Dense(Layer):
    def __init__(self, N, d_in, d_out, d=1, activation='id'):
        self.activation_layer = None
        if activation == 'relu':
            self.activation_layer = Relu([N, d, d_out])

        self.N = N
        self.d_in = d_in
        self.d_out = d_out
        self.d = d

        self.X = Tensor([N, d, d_in], sfix)
        self.Y = Tensor([N, d, d_out], sfix)
        self.W = Tensor([d_out, d_in], sfix)
        # self.W.alloc()
        self.b = sfix.Array(d_out)
        self.b.assign_all(0)

        back_N = N
        self.nabla_Y = Tensor([back_N, d, d_out], sfix)
        self.nabla_X = Tensor([back_N, d, d_in], sfix)
        self.nabla_W = sfix.Matrix(d_out, d_in)
        self.nabla_W.alloc()
        
        self.nabla_W_t = self.nabla_W.transpose()
        
        self.nabla_b = sfix.Array(d_out)
        
        self.nabla_W_per_sample_norms = sfix.Array(Optimizer.batch_size)
        self.nabla_b_per_sample_norms = sfix.Array(Optimizer.batch_size)
        self.nabla_b_clipped_sum = sfix.Array(self.d_out)
                
        if self.activation_layer:
            self.f_input = self.activation_layer.X
            self.activation_layer.Y = self.Y
            self.activation_layer.nabla_Y = self.nabla_Y
        else:
            self.f_input = self.Y
            
        self.noise_b = sfix.Array(self.d_out)
        self.noise_W = sfix.Array(self.d_out*self.d_in)
                                                
    def reset(self):
        d_in = self.d_in
        d_out = self.d_out
        r = (math.sqrt(6.0 / (d_in + d_out)))
        print('Initializing dense weights in [%f,%f]' % (-r, r))
        self.W.randomize(-r, r)
        self.b.assign_all(0)
                        
    def _forward(self, batch=None):
        N = len(batch)
        # if batch is None:
        #     batch = regint.Array(self.N)
        #     batch.assign(regint.inc(self.N))
                    
        assert self.d == 1
        if self.input_bias:
            prod = MultiArray([N, self.d, self.d_out], sfix)
        else:
            prod = self.f_input
            
        max_size = get_program().budget // self.d_out
        
        ##### THIS IS self.N immune
        ##### THIS PRODUCT IS ALWAYS ASSIGNED TO THE FIRST "batch_size" ELEMENTS OF THE OUTPUT VECTOR (REST ALL ARE 0!)
        offset = MemValue(self.optimizer.iterations_done*N) if not self.compute_nabla_X else MemValue(0)
        @multithread(self.n_threads, N, max_size)
        def _(base, size):
            X_sub = sfix.Matrix(self.N, self.d_in, address=self.X.address)
            prod.assign_part_vector(X_sub.direct_mul_trans(self.W, indices=(
                                            batch.get_vector(base, size), 
                                            regint.inc(self.d_in),
                                            regint.inc(self.d_in), 
                                            regint.inc(self.d_out)
                                        )
                                    ), base)

        
        @for_range_multithread(self.n_threads, 1, N)
        def _(i):
            v = prod[i].get_vector() + self.b.get_vector()
            self.f_input[i].assign_vector(v)
                    
        # print_ln("W = %s", self.W[0].get_vector(0, 100).reveal())
        # print_ln("X = %s", self.X.get_vector(0, 100).reveal())
        # print_ln("B = %s", self.b.get_vector(0, min(100, self.b.total_size())).reveal())
        # print_ln("Y = %s", self.f_input[0].get_vector(0, min(100, self.f_input[0].total_size())).reveal())
        
        if self.activation_layer:
            self.activation_layer._forward(batch)
        
    def _backward(self, batch=None):
        if self.activation_layer:
            self.activation_layer._backward(batch)
            self.nabla_Y = self.activation_layer.nabla_X
        else:
            self.nabla_Y = self.nabla_Y # [N x d_out]   
        
        N = len(batch)
        if self.compute_nabla_X:
            # print_ln("W = %s", self.W[0].get_vector(0, 100).reveal())    
            # print_ln("NABLA_Y = %s", self.nabla_Y.get_vector(0, 100).reveal()) 
            
            self.nabla_X.alloc()
            @multithread(self.n_threads, N)
            def _(base, size):
                B = sfix.Matrix(N, self.d_out, address=self.nabla_Y.address) # [N x d_out]
                self.nabla_X.assign_part_vector(
                    B.direct_mul(self.W, indices=(regint.inc(size, base),
                                                   regint.inc(self.d_out),
                                                   regint.inc(self.d_out),
                                                   regint.inc(self.d_in))),
                    base)
            
            # print_ln("NABLA_X = %s", self.nabla_X.get_vector(0, 100).reveal())
        
        if self.optimizer.clip:
            self.accumulate_per_sample_norms(batch)
        else:
            self._backward_params(batch)

    def _backward_params(self, batch):
        N = len(batch)
        tmp = Matrix(self.d_in, self.d_out, unreduced_sfix)

        A = sfix.Matrix(N, self.d_out, address=self.nabla_Y.address)
        B = sfix.Matrix(self.N, self.d_in, address=self.X.address)

        @multithread(self.n_threads, self.d_in)
        def _(base, size):
            mp = B.direct_trans_mul(A, reduce=False,
                                    indices=(regint.inc(size, base),
                                             batch.get_vector(),
                                             regint.inc(N),
                                             regint.inc(self.d_out)))
            tmp.assign_part_vector(mp, base)
            
        self.nabla_W_t = self.nabla_W.transpose()
            
        @multithread(self.n_threads, self.d_in * self.d_out,
                     max_size=get_program().budget)
        def _(base, size):
            self.nabla_W_t.assign_vector(
                tmp.get_vector(base, size).reduce_after_mul(), base=base)
            
        self.nabla_W.assign(self.nabla_W_t.transpose())
            
        self.nabla_b.assign_vector(sum(sum(self.nabla_Y[k][j].get_vector()
                                    for k in range(N))
                                for j in range(self.d)))
        
        
        # print_ln("NABLA_W = %s", self.nabla_W[0].get_vector(0, 100).reveal())
        # print_ln("NABLA_B = %s", self.nabla_b.get_vector(0, min(self.nabla_b.total_size(), 100)).reveal())                                 
    
    def zero_grad(self):
        self.nabla_W.assign_all(0)
        self.nabla_b.assign_all(0)
   
    def accumulate_per_sample_norms(self, batch):
        N = len(batch)
        
        # FOR BIAS 
        assert N == self.nabla_b_per_sample_norms.total_size()
        @for_range_multithread(self.n_threads, 1, N)
        def _(i):
            nabla_Y_i = self.nabla_Y[i][0].get_vector()   
            L2_norm_sq = get_L2_norm_sq(nabla_Y_i)
            self.nabla_b_per_sample_norms[i] = (L2_norm_sq)
        
        # FOR WEIGHTS
        assert N == self.nabla_W_per_sample_norms.total_size()
        A = sfix.Matrix(N, self.d_out, address=self.nabla_Y.address)
        B = sfix.Matrix(self.N, self.d_in, address=self.X.address)

        #### THIS HAS BEEN MADE self.N IMMUNE
        offset = MemValue(self.optimizer.iterations_done*N) if not self.compute_nabla_X else MemValue(0)
        #selector = regint.inc(N) if self.compute_nabla_X else batch
        @for_range_multithread(self.n_threads, 1, N)
        def _(i):
            # vec_c = B[i + offset].get_vector()
            vec_c = B[batch[i]].get_vector()
            sum_vec_c_2 = get_L2_norm_sq(vec_c)
            
            vec_r = A[i].get_vector()
            sum_vec_r_2 = get_L2_norm_sq(vec_r)

            L2_norm_sq = sum_vec_c_2 * sum_vec_r_2
            self.nabla_W_per_sample_norms[i] = (L2_norm_sq)
      
    def get_noise(self, mode):
        if mode == "weight":
            self.noise_W.input_from(0)
            return self.noise_W
        else:
            self.noise_b.input_from(0)
            return self.noise_b
      
    def _clip_and_accumulate(self, batch, clipping_multiplier):
        N = len(batch)
        tmp = Matrix(self.d_in, self.d_out, unreduced_sfix)
        
        self.nabla_b_pre = sfix.Array(self.d_out)
        self.nabla_b_pre.assign_vector(sum(sum(self.nabla_Y[k][j].get_vector() * (clipping_multiplier[k]).expand_to_vector(self.d_out)
                                           for k in range(N))
                                       for j in range(self.d)))
        
        
        self.nabla_b.assign(self.nabla_b_pre)

        ## WEIGHT CLIPPING ##
        A = sfix.Matrix(N, self.d_out, address=self.nabla_Y.address)
        B = sfix.Matrix(self.N, self.d_in, address=self.X.address)
        
        @for_range_opt_multithread(self.n_threads, N)
        def _(i):
            vec_r = A[i].get_vector()
            vec_r = vec_r * clipping_multiplier[i]
            A[i].assign_vector(vec_r)

        ##### THIS HAS BEEN MADE self.N IMMUNE ##### 
        offset = MemValue(self.optimizer.iterations_done*N) if not self.compute_nabla_X else MemValue(0)
        # selector = regint.inc(N) if self.compute_nabla_X else batch
        @multithread(self.n_threads, self.d_in)
        def _(base, size):
            mp = B.direct_trans_mul(A, reduce = False,
                                        indices = (
                                            regint.inc(size, base),
                                            # regint.inc(N, offset),
                                            batch.get_vector(),
                                            regint.inc(N),
                                            regint.inc(self.d_out)
                                        )
                                    )
            tmp.assign_part_vector(mp, base)
            
        @multithread(self.n_threads, self.d_in * self.d_out, max_size=get_program().budget)
        def _(base, size):
            self.nabla_W_t.assign_vector(
                tmp.get_vector(base, size).reduce_after_mul(), base=base)
            
        self.nabla_W.assign(self.nabla_W_t.transpose())
           
        if self.optimizer.noisy:
            n = self.nabla_b.total_size()
            noise = self.optimizer.noise_bag.get_vector(self.optimizer.noise_bag_cursor, n)
            self.optimizer.noise_bag_cursor += n
            # print_ln("NOISE = %s", noise[:10].reveal())
            self.nabla_b.assign_vector(self.nabla_b.get_vector() + noise)
            
            n = self.nabla_W.total_size()
            noise = self.optimizer.noise_bag.get_vector(self.optimizer.noise_bag_cursor, n)
            self.optimizer.noise_bag_cursor += n
            self.nabla_W.assign_vector(self.nabla_W.get_vector() + noise)            

class FixBase:
    bias_before_reduction = False

    @staticmethod
    def new_squant():
        class _(sfix):
            params = None
        return _

    def input_params_from(self, player):
        pass

    def const_div(self, acc, n):
        return (sfix._new(acc) * self.output_squant(1 / n)).v

class BaseLayer(Layer):
    def __init__(self, input_shape, output_shape, inputs=None):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.input_squant = self.new_squant()
        self.output_squant = self.new_squant()

        self.X = Tensor(input_shape, self.input_squant)
        self.Y = Tensor(output_shape, self.output_squant)

        back_shapes = list(input_shape), list(output_shape)
        for x in back_shapes:
            x[0] = min(x[0], self.back_batch_size)

        self.nabla_X = Tensor(back_shapes[0], self.input_squant)
        self.nabla_Y = Tensor(back_shapes[1], self.output_squant)
        self.inputs = inputs

    def temp_shape(self):
        return [0]

    @property
    def N(self):
        return self.input_shape[0]

class ConvBase(BaseLayer):
    fewer_rounds = True
    use_conv2ds = True
    temp_weights = None
    temp_inputs = None
    thetas = lambda self: (self.weights, self.bias)
    nablas = lambda self: (self.nabla_weights, self.nabla_bias)

    @classmethod
    def init_temp(cls, layers):
        size = 0
        for layer in layers:
            size = max(size, reduce(operator.mul, layer.temp_shape()))
        cls.temp_weights = sfix.Array(size)
        cls.temp_inputs = sfix.Array(size)

    def __init__(self, input_shape, weight_shape, bias_shape, output_shape, stride,
                 padding='SAME', tf_weight_format=False, inputs=None):
        super(ConvBase, self).__init__(input_shape, output_shape, inputs=inputs)

        self.weight_shape = weight_shape
        self.bias_shape = bias_shape
        self.stride = stride
        self.tf_weight_format = tf_weight_format
        if padding == 'SAME':
            # https://web.archive.org/web/20171223022012/https://www.tensorflow.org/api_guides/python/nn
            self.padding = []
            for i in 1, 2:
                s = stride[i - 1]
                assert output_shape[i] >= input_shape[i] // s
                if tf_weight_format:
                    w = weight_shape[i - 1]
                else:
                    w = weight_shape[i]
                if (input_shape[i] % stride[1] == 0):
                    pad_total = max(w - s, 0)
                else:
                    pad_total = max(w - (input_shape[i] % s), 0)
                self.padding.append(pad_total // 2)
        elif padding == 'VALID':
            self.padding = [0, 0]
        elif isinstance(padding, int):
            self.padding = [padding, padding]
        else:
            self.padding = padding

        self.weight_squant = self.new_squant()
        self.bias_squant = self.new_squant()

        self.weights = Tensor(weight_shape, self.weight_squant)
        self.bias = Array(output_shape[-1], self.bias_squant)

        self.nabla_weights = Tensor(weight_shape, self.weight_squant)
        self.nabla_bias = Array(output_shape[-1], self.bias_squant)

        self.unreduced = Tensor(self.output_shape, sint, address=self.Y.address)

        if tf_weight_format:
            weight_in = weight_shape[2]
        else:
            weight_in = weight_shape[3]
        assert(weight_in == input_shape[-1])
        assert(bias_shape[0] == output_shape[-1])
        assert(len(bias_shape) == 1)
        assert(len(input_shape) == 4)
        assert(len(output_shape) == 4)
        assert(len(weight_shape) == 4)

    def __repr__(self):
        return '%s(%s, %s, %s, %s, %s, padding=%s, tf_weight_format=%s)' % \
            (type(self).__name__, self.X.sizes, self.weight_shape,
             self.bias_shape, self.Y.sizes, self.stride, repr(self.padding),
             self.tf_weight_format)

    def input_from(self, player, **kwargs):
        self.input_params_from(player)
        self.weights.input_from(player, budget=100000, **kwargs)
        if self.input_bias:
            self.bias.input_from(player, **kwargs)

    def output_weights(self):
        self.weights.print_reveal_nested()
        print_ln('%s', self.bias.reveal_nested())

    def reveal_parameters_to_binary(self):
        assert not self.tf_weight_format
        n_filters = self.weights.shape[0]
        n_channels = self.weights.shape[3]
        @for_range(n_filters)
        def _(i):
            @for_range(n_channels)
            def _(j):
                part = self.weights.get_vector_by_indices(i, None, None, j)
                part.reveal().binary_output()
        self.bias.reveal_to_binary_output()

    def dot_product(self, iv, wv, out_y, out_x, out_c):
        bias = self.bias[out_c]
        acc = self.output_squant.unreduced_dot_product(iv, wv)
        acc.v += bias.v
        acc.res_params = self.output_squant.params
        #self.Y[0][out_y][out_x][out_c] = acc.reduce_after_mul()
        self.unreduced[0][out_y][out_x][out_c] = acc.v

    def reduction(self, batch_length=1):
        unreduced = self.unreduced
        n_summands = self.n_summands()
        #start_timer(2)
        n_outputs = batch_length * reduce(operator.mul, self.output_shape[1:])
        @multithread(self.n_threads, n_outputs,
                     1000 if sfix.round_nearest else 10 ** 6)
        def _(base, n_per_thread):
            res = self.input_squant().unreduced(
                sint.load_mem(unreduced.address + base,
                              size=n_per_thread),
                self.weight_squant(),
                self.output_squant.params,
                n_summands).reduce_after_mul()
            res.store_in_mem(self.Y.address + base)
        #stop_timer(2)

    def temp_shape(self):
        return list(self.output_shape[1:]) + [self.n_summands()]

    def prepare_temp(self):
        shape = self.temp_shape()
        inputs = MultiArray(shape, self.input_squant,
                            address=self.temp_inputs)
        weights = MultiArray(shape, self.weight_squant,
                             address=self.temp_weights)
        return inputs, weights

class Conv2d(ConvBase):
    def n_summands(self):
        _, weights_h, weights_w, _ = self.weight_shape
        _, inputs_h, inputs_w, n_channels_in = self.input_shape
        return weights_h * weights_w * n_channels_in

    def _forward(self, batch):
        if self.tf_weight_format:
            assert(self.weight_shape[3] == self.output_shape[-1])
            weights_h, weights_w, _, _ = self.weight_shape
        else:
            assert(self.weight_shape[0] == self.output_shape[-1])
            _, weights_h, weights_w, _ = self.weight_shape
        _, inputs_h, inputs_w, n_channels_in = self.input_shape
        _, output_h, output_w, n_channels_out = self.output_shape

        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding

        if self.use_conv2ds:
            part_size = 1
            @for_range_opt_multithread(self.n_threads,
                                       [len(batch), n_channels_out])
            def _(i, j):
                inputs = self.X.get_slice_vector(
                    batch.get_part(i * part_size, part_size))
                if self.tf_weight_format:
                    weights = self.weights.get_vector_by_indices(None, None, None, j)
                else:
                    weights = self.weights.get_part_vector(j)
                inputs = inputs.pre_mul()
                weights = weights.pre_mul()
                res = sint(size = output_h * output_w * part_size)
                conv2ds(res, inputs, weights, output_h, output_w,
                        inputs_h, inputs_w, weights_h, weights_w,
                        stride_h, stride_w, n_channels_in, padding_h, padding_w,
                        part_size)
                if self.bias_before_reduction:
                    res += self.bias.expand_to_vector(j, res.size).v
                else:
                    res += self.bias.expand_to_vector(j, res.size).v << \
                        self.input_squant.f
                addresses = regint.inc(res.size,
                                       self.unreduced[i * part_size].address + j,
                                       n_channels_out)
                res.store_in_mem(addresses)
            self.reduction(len(batch))
            if self.debug_output:
                print_ln('%s weights %s', self, self.weights.reveal_nested())
                print_ln('%s bias %s', self, self.bias.reveal_nested())
                @for_range(len(batch))
                def _(i):
                    print_ln('%s X %s %s', self, i, self.X[batch[i]].reveal_nested())
                    print_ln('%s Y %s %s', self, i, self.Y[i].reveal_nested())
            return
        else:
            assert len(batch) == 1
            if self.fewer_rounds:
                inputs, weights = self.prepare_temp()

        @for_range_opt_multithread(self.n_threads,
                                   [output_h, output_w, n_channels_out])
        def _(out_y, out_x, out_c):
                    in_x_origin = (out_x * stride_w) - padding_w
                    in_y_origin = (out_y * stride_h) - padding_h
                    iv = []
                    wv = []
                    for filter_y in range(weights_h):
                        in_y = in_y_origin + filter_y
                        inside_y = (0 <= in_y) * (in_y < inputs_h)
                        for filter_x in range(weights_w):
                            in_x = in_x_origin + filter_x
                            inside_x = (0 <= in_x) * (in_x < inputs_w)
                            inside = inside_y * inside_x
                            if is_zero(inside):
                                continue
                            for in_c in range(n_channels_in):
                                iv += [self.X[0][in_y * inside_y]
                                       [in_x * inside_x][in_c]]
                                wv += [self.weights[out_c][filter_y][filter_x][in_c]]
                                wv[-1] *= inside
                    if self.fewer_rounds:
                        inputs[out_y][out_x][out_c].assign(iv)
                        weights[out_y][out_x][out_c].assign(wv)
                    else:
                        self.dot_product(iv, wv, out_y, out_x, out_c)

        if self.fewer_rounds:
            @for_range_opt_multithread(self.n_threads,
                                       list(self.output_shape[1:]))
            def _(out_y, out_x, out_c):
                self.dot_product(inputs[out_y][out_x][out_c],
                                 weights[out_y][out_x][out_c],
                                 out_y, out_x, out_c)

        self.reduction()

class FixConv2d(Conv2d, FixBase):
    """ Fixed-point 2D convolution layer.

    :param input_shape: input shape (tuple/list of four int)
    :param weight_shape: weight shape (tuple/list of four int)
    :param bias_shape: bias shape (tuple/list of one int)
    :param output_shape: output shape (tuple/list of four int)
    :param stride: stride (tuple/list of two int)
    :param padding: :py:obj:`'SAME'` (default), :py:obj:`'VALID'`, or tuple/list of two int
    :param tf_weight_format: weight shape format is (height, width, input channels, output channels) instead of the default (output channels, height, width, input channels)
    """

    def reset(self):
        assert not self.tf_weight_format
        n_in = reduce(operator.mul, self.weight_shape[1:])
        r = math.sqrt(6.0 / (n_in + self.weight_shape[0]))
        print('Initializing convolution weights in [%f,%f]' % (-r, r))
        self.weights.randomize(-r, r, n_threads=self.n_threads)
        self.bias.assign_all(0)

    def backward(self, compute_nabla_X=True, batch=None):
        assert self.use_conv2ds

        assert not self.tf_weight_format
        _, weights_h, weights_w, _ = self.weight_shape
        _, inputs_h, inputs_w, n_channels_in = self.input_shape
        _, output_h, output_w, n_channels_out = self.output_shape

        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding

        N = len(batch)

        self.nabla_bias.assign_all(0)

        @for_range(N)
        def _(i):
            self.nabla_bias.assign_vector(
                self.nabla_bias.get_vector() + sum(sum(
                    self.nabla_Y[i][j][k].get_vector() for k in range(output_w))
                                                   for j in range(output_h)))

        input_size = inputs_h * inputs_w * N
        batch_repeat = regint.Matrix(N, inputs_h * inputs_w)
        batch_repeat.assign_vector(batch.get(
            regint.inc(input_size, 0, 1, 1, N)) *
                                   reduce(operator.mul, self.input_shape[1:]))

        @for_range_opt_multithread(self.n_threads, [n_channels_in, n_channels_out])
        def _(i, j):
            a = regint.inc(input_size, self.X.address + i, n_channels_in, N,
                           inputs_h * inputs_w)
            inputs = sfix.load_mem(batch_repeat.get_vector() + a).pre_mul()
            b = regint.inc(N * output_w * output_h, self.nabla_Y.address + j, n_channels_out, N)
            rep_out = regint.inc(output_h * output_w * N, 0, 1, 1, N) * \
                reduce(operator.mul, self.output_shape[1:])
            nabla_outputs = sfix.load_mem(rep_out + b).pre_mul()
            res = sint(size = weights_h * weights_w)
            conv2ds(res, inputs, nabla_outputs, weights_h, weights_w, inputs_h,
                    inputs_w, output_h, output_w, -stride_h, -stride_w, N,
                    padding_h, padding_w, 1)
            reduced = unreduced_sfix._new(res).reduce_after_mul()
            self.nabla_weights.assign_vector_by_indices(reduced, j, None, None, i)

        if compute_nabla_X:
            assert tuple(self.stride) == (1, 1)
            reverse_weights = MultiArray(
                [n_channels_in, weights_h, weights_w, n_channels_out], sfix)
            @for_range_opt_multithread(self.n_threads, n_channels_in)
            def _(l):
                @for_range(weights_h)
                def _(j):
                    @for_range(weights_w)
                    def _(k):
                        addresses = regint.inc(n_channels_out,
                            self.weights[0][j][weights_w-k-1].get_address(l),
                            reduce(operator.mul, self.weights.sizes[1:]))
                        reverse_weights[l][weights_h-j-1][k].assign_vector(
                            self.weights.value_type.load_mem(addresses))
            padded_w = inputs_w + 2 * padding_w
            padded_h = inputs_h + 2 * padding_h
            if padding_h or padding_w:
                output = MultiArray(
                    [N, padded_h, padded_w, n_channels_in], sfix)
            else:
                output = self.nabla_X
            @for_range_opt_multithread(self.n_threads,
                                       [N, n_channels_in])
            def _(i, j):
                res = sint(size = (padded_w * padded_h))
                conv2ds(res, self.nabla_Y[i].get_vector().pre_mul(),
                        reverse_weights[j].get_vector().pre_mul(),
                        padded_h, padded_w, output_h, output_w,
                        weights_h, weights_w, 1, 1, n_channels_out,
                        weights_h - 1, weights_w - 1, 1)
                output.assign_vector_by_indices(
                    unreduced_sfix._new(res).reduce_after_mul(),
                    i, None, None, j)
            if padding_h or padding_w:
                @for_range_opt_multithread(self.n_threads, N)
                def _(i):
                    @for_range(inputs_h)
                    def _(j):
                        @for_range(inputs_w)
                        def _(k):
                            jj = j + padding_w
                            kk = k + padding_w
                            self.nabla_X[i][j][k].assign_vector(
                                output[i][jj][kk].get_vector())

        if self.debug_output:
            @for_range(len(batch))
            def _(i):
                print_ln('%s X %s %s', self, i, list(self.X[i].reveal_nested()))
                print_ln('%s nabla Y %s %s', self, i, list(self.nabla_Y[i].reveal_nested()))
                if compute_nabla_X:
                    print_ln('%s nabla X %s %s', self, i, self.nabla_X[batch[i]].reveal_nested())
            print_ln('%s nabla weights %s', self,
                     (self.nabla_weights.reveal_nested()))
            print_ln('%s weights %s', self, (self.weights.reveal_nested()))
            print_ln('%s nabla b %s', self, (self.nabla_bias.reveal_nested()))
            print_ln('%s bias %s', self, (self.bias.reveal_nested()))

class MultiOutput(Layer):
    def __init__(self, N, d_out):
        self.X = sfix.Matrix(N, d_out)
        self.Y = sint.Matrix(N, d_out)
        self.nabla_X = sfix.Matrix(N, d_out)
        self.l = MemValue(sfix(-1))
        self.losses = sfix.Array(N)
        self.N = N
        self.d_out = d_out
        self.compute_loss = True
        self.exp = sfix.Matrix(N, d_out)
        self.positives = sint.Matrix(N, d_out)
        self.relus = sfix.Matrix(N, d_out)
        self.true_X = sfix.Array(N)
        
        self.sum_exp = sfix.Array(N)
        
    def _forward(self, batch):
        N = len(batch)
        tmp = self.losses
        @for_range_opt_multithread(self.n_threads, N)
        def _(i):
            e, m = exp_for_softmax(self.X[i])
            self.exp[i].assign_vector(e)

            true_X = sfix.dot_product(self.Y[batch[i]], self.X[i])
            tmp[i] = m + log_e(sum(e)) - true_X
            self.true_X[i] = true_X  
        
        self.l.write(sum(tmp.get_vector(0, N)) / N)
        print_ln("Loss = %s", self.l.reveal())
        
        # self.__forward(batch)
               
    def _backward(self, batch):
        d_out = self.X.sizes[1]
        @for_range_opt_multithread(self.n_threads, len(batch))
        def _(i):
            div = softmax_from_exp(self.exp[i])
            self.nabla_X[i][:] = -self.Y[batch[i]][:] + div

        # print_ln("NABLA_X_0__ = %s", self.nabla_X[0][:100].reveal())        
                       
    def reveal_correctness(self, n=None, Y=None, debug=False):
        if n is None:
            n = self.X.sizes[0]
        if Y is None:
            Y = self.Y

        assert n <= len(self.X)
        assert n <= len(Y)
        Y.address = MemValue.if_necessary(Y.address)
        @map_sum(None if debug else self.n_threads, None, n, 1, regint)
        def _(i):
            a = Y[i].reveal_list()
            b = self.X[i].reveal_list()
            truth = argmax(a)
            guess = argmax(b)
            correct = truth == guess
            return correct
        return _()

class SGD(Optimizer):
    dataset = ""
    
    def __init__(self, layers: List[Layer], input_params = False, lr: float = 0.1, clip=0, sigma=0, double_noise=False, min_lr: float = 0.1, lr_red_max_epoch: int = 10):
        self.layers = layers
        self.learnable_layers = self.layers[:-1]
        self.nablas = []
        self.thetas = []
        self.per_sample_norms = []
        self.delta_thetas = []
        self.lr = lr 
        self.epoch_num = MemValue(0)
        self.X_by_label = None
        self.shuffle = False
        self.momentum = 0
        self.early_division = False    
        self.debug = True
        
        self.total_params = 0
        self.aggregated_norms = sfix.Array(Optimizer.batch_size)
        self.clip = clip 
        self.noisy = sigma
        self.double_noise = double_noise
        self.min_lr = min_lr
        self.lr_red_max_epoch = lr_red_max_epoch
        self.lr_step = (self.min_lr - self.lr)/(self.lr_red_max_epoch-1) if self.lr_red_max_epoch > 1 else 0
        
        prev_layer = None
        for i, layer in enumerate(self.layers):
            layer.optimizer = self
            if i != len(layers)-1:
                # all but the last layer
                if not input_params:
                    layer: Dense
                    layer.reset()
                
                self.thetas.extend(layer.thetas())
                self.nablas.extend(layer.nablas())
                self.per_sample_norms.extend(layer.per_sample_norms())
                for theta in self.thetas:
                    self.delta_thetas.append(theta.same_shape())                
                    
            if i > 0 and prev_layer is not None:
                # all but the first layer
                layer.inputs = [prev_layer]
                layer.compute_nabla_X = True
            
            prev_layer = layer
            
        for layer in self.learnable_layers:
            layer: Dense
            self.total_params += layer.W.total_size()
            self.total_params += layer.b.total_size()
            
        self.noise_bag = sfix.Array(self.total_params)
        print(f"Setting up noise reception for {self.total_params} samples")
        
        if self.noisy:
            @if_(get_player_id()._v == 0)
            def _():
                print_ln("Setting up connection %s", get_player_id()._v.reveal())
                global socket
                listen_for_clients(15000)
                socket = accept_client_connection(15000)   
            
            @if_(get_player_id()._v == 1)
            def _():
                print_ln("Setting up connection %s", get_player_id()._v.reveal())
                global socket
                listen_for_clients(15000)
                socket = accept_client_connection(15000)   
                                            
    def batch_for(self, layer, batch):
        if layer in (self.layers[0], self.layers[-1]):
            return batch
        else:
            batch = regint.Array(len(batch))
            batch.assign(regint.inc(len(batch)))
            return batch    
        
    def forward(self, batch_size, batch = None):
        N = batch_size
        if batch is None:
            batch = regint.Array(N)
            batch.assign(regint.inc(N))
            
        for i, layer in enumerate(self.layers):
            if layer.inputs and len(layer.inputs) > 0:
                # all but the first layer
                layer._X.address = layer.inputs[0].Y.address
                
            layer.Y.alloc()
            layer.forward(batch=self.batch_for(layer, batch))
            
    def backward(self, batch):
        for i, layer in reversed(list(enumerate(self.layers))):
            assert len(batch) <= layer.back_batch_size
            if not layer.inputs:
                # First layer
                layer._backward(batch=self.batch_for(layer, batch))
            else:
                layer.compute_nabla_X = True
                layer._backward(batch=self.batch_for(layer, batch))
                layer.inputs[0].nabla_Y.address = layer.nabla_X.address
    
    def calculate_eff_clip_multipliers(self, batch):
        N = len(batch)
        
        self.aggregated_norms.assign_all(0)
        for p in self.per_sample_norms:
            self.aggregated_norms.assign_vector(self.aggregated_norms.get_vector() + p.get_vector())
            
        self.aggregated_norms.assign_vector(mpc_math.sqrt(self.aggregated_norms.get_vector()))
        
        comparision: sfix = self.aggregated_norms.get_vector() > sfix(self.clip).expand_to_vector(N)
        
        self.eff_clipping_multiplier = sfix.Array(N)
        self.eff_clipping_multiplier.assign_vector(comparision.if_else(sfix(self.clip).expand_to_vector(N)/self.aggregated_norms.get_vector(), 1))
    
    def pre_update_clip_step(self, batch):
        assert self.clip and self.clip > 0
        
        self.calculate_eff_clip_multipliers(batch)
        for layer in reversed(self.learnable_layers):
            layer: Dense 
            layer._clip_and_accumulate(self.batch_for(layer, batch), self.eff_clipping_multiplier)
            
            # noise = self.get_noise_from_file(layer.d_out)
            # layer.nabla_b.assign_vector(layer.nabla_b.get_vector() + noise)
            
    def update(self, batch):    
        N = len(batch)    
        if self.clip:
            self.pre_update_clip_step(batch)

        for nabla, theta, delta_theta in zip(self.nablas, self.thetas, self.delta_thetas):
            if 0:
                if type(nabla) == Matrix:
                    print_ln("THETA_W = %s", (theta[0].get_vector(0, min(nabla[0].total_size(), 100))).reveal())
                else:
                    print_ln("THETA_B = %s", theta.get_vector(0, min(theta.total_size(), 10)).reveal())     
            
            if 0:
                if type(nabla) == Matrix:
                    print_ln("NABLA_W = %s", (nabla[0].get_vector(0, min(nabla[0].total_size(), 10))).reveal())
                else:
                    print_ln("NABLA_B = %s", nabla.get_vector(0, 10).reveal())                                 

            @multithread(self.n_threads, nabla.total_size())
            def _(base, size):
                nabla_vector = nabla.get_vector(base, size)
                
                new = nabla_vector*(sfix(-self.update_factor).expand_to_vector(size))
                delta_theta.assign_vector(new, base)
                theta.assign_vector(theta.get_vector(base, size) + nabla.get_vector(base, size)*sfix(-self.update_factor), base)
                
            if 0:
                if type(nabla) == Matrix:
                    print_ln("NABLA_W_SCALED = %s", (nabla[0].get_vector(0, min(nabla[0].total_size(), 1000))*sfix(-self.update_factor)).reveal())
                else:
                    print_ln("NABLA_B_SCALED = %s", (nabla.get_vector(0, min(nabla.total_size(), 1000))*sfix(-self.update_factor)).reveal())        

        for layer in self.layers[:-1]:
            layer: Dense
            layer.zero_grad()
    
    def save_params(self):
        print_ln("Saving model parameters...")
        sfix(self.epoch_num).write_to_file(0)
        sfix(self.iterations_done).write_to_file()
        for layer in self.learnable_layers:
            layer: Dense 
            layer.W.write_to_file()
            layer.b.write_to_file()
            
    def print_training_config(self, batch_size, n_epochs):
        print_both("***********************************************************")
        print_both(f"Training {SGD.dataset}")
        print_both("Model: ", end="")
        for layer in self.layers:
            print_both(layer.__repr__(), end=" => ")
        print_both(f"\nTrain Examples: {self.layers[0].N}")
        print_both(f"Batch Size: {batch_size}")
        print_both(f"Num Epochs: {n_epochs}")
        print_both(f"Learning Rate: {self.lr} to {self.min_lr} over {self.lr_red_max_epoch} epochs")
        if self.clip:
            print_both(f"Clipping Factor: {self.clip}")
        if self.noisy:
            print_both(f"Sigma: {self.noisy}")

        print_both("***********************************************************")
    
    def reveal_correctness(self, data, truth, batch_size=128, running=False):
        """ Test correctness by revealing results.

        :param data: test sample data
        :param truth: test labels
        :param batch_size: batch size
        :param running: output after every batch

        """
        N = data.sizes[0]
        n_correct = MemValue(0)
        loss = MemValue(sfix(0))
        def f(start, batch_size, batch):
            batch.assign_vector(regint.inc(batch_size, start))
            self.forward(batch_size=batch_size, batch=batch)
            part_truth = truth.get_part(start, batch_size)
            n_correct.iadd(
                self.layers[-1].reveal_correctness(batch_size, part_truth))
            loss.iadd(self.layers[-1].l * batch_size)
            if running:
                total = start + batch_size
                print_str('\rpart acc: %s (%s/%s) ',
                          cfix(n_correct, k=63, f=31) / total, n_correct, total)
        self.run_in_batches(f, data, batch_size, truth)
        if running:
            print_ln()
        loss = loss.reveal()
        if cfix.f < 31:
            loss = cfix._new(loss.v << (31 - cfix.f), k=63, f=31)
        return n_correct, loss / N

    def run_in_batches(self, f, data, batch_size, truth=None):
        batch_size = min(batch_size, data.sizes[0])
        training_data = self.layers[0].X.address
        training_truth = self.layers[-1].Y.address
        self.layers[0].X.address = data.address
        if truth:
            self.layers[-1].Y.address = truth.address
        N = data.sizes[0]
        batch = regint.Array(batch_size)
        @for_range(N // batch_size)
        def _(i):
            start = i * batch_size
            f(start, batch_size, batch)
        batch_size = N % batch_size
        if batch_size:
            start = N - batch_size
            f(start, batch_size, regint.Array(batch_size))
        self.layers[0].X.address = training_data
        self.layers[-1].Y.address = training_truth
    
    def test(self, batch_size, test_X, test_Y):
        n_test = len(test_Y)
        n_correct, loss = self.reveal_correctness(test_X, test_Y, batch_size)
        print_ln('TEST LOSS = %s', loss)        
        print_ln('TEST ACC = %s % (%s/%s)', ((self.n_correct/n_test)*100).reveal(), n_correct.reveal(), n_test)        

    def train(self, batch_size, n_epochs, test_loader = None):
        self.print_stats = True
        
        assert batch_size > 0
            
        self.lr = MemValue(cfix(self.lr))
        self.n_correct = MemValue(0)
        self.counter = 0
        @for_range_opt(n_epochs)
        def _(i):            
            self.epoch_num.iadd(1)
            
            if self.X_by_label is None:
                self.X_by_label = [[None] * self.layers[0].N]   # [60000]

            n_iterations = self.layers[0].N // batch_size
            print('%d runs per epoch' % n_iterations)
            
            indices_by_label = []
            X = self.X_by_label[0]
            indices = regint.Array(batch_size * n_iterations)   # [60000]
            indices_by_label.append(indices)
            
            indices.assign(regint.inc(len(X)))
            indices.shuffle()
            
            self.n_correct.write(0)
            train_loss = MemValue(sfix(0))
            self.iterations_done = MemValue(0)
            self.update_factor = MemValue(self.lr / self.batch_size)
                            
            # TRAIN EACH BATCH
            @for_range_opt(n_iterations)
            def _(j): 
                if self.noisy:
                    @if_(get_player_id()._v >= 0)
                    def _():
                        self.get_noise_samples_from_worker_file(self.total_params)
                
                print_str("Epoch %s.%s: ", i+1, j+1)
                
                batch = regint.Array(batch_size)
                    
                indices = indices_by_label[0]                
                batch.assign(indices.get_vector(j*batch_size, batch_size))
                                
                self.forward(batch_size=batch_size, batch=batch)
                self.backward(batch=batch)
                self.update(batch)
                
                self.iterations_done.iadd(1)
                
                # CALCULATE TRAIN STATS ###
                train_loss.iadd(self.layers[-1].l)
                batch_targets = self.layers[-1].Y.same_shape()
                batch_targets.assign_vector(self.layers[-1].Y.get_slice_vector(batch))
                self.n_correct.iadd(self.layers[-1].reveal_correctness(batch_size, batch_targets))
            
            if self.print_stats:
                print_ln("TRAIN LOSS = %s", (train_loss/n_iterations).reveal())
                print_ln("TRAIN ACC = %s % (%s/%s)", ((sfix(self.n_correct, k=31, f=16)/self.layers[0].N)*100).reveal(),  self.n_correct.reveal(), self.layers[0].N)
            
            if test_loader:
                test_X, test_Y = test_loader
                self.test(batch_size, test_X, test_Y)

            @if_(i < self.lr_red_max_epoch)
            def _():
                self.lr += self.lr_step
                print_ln("Reducing learning rate to %s", self.lr.reveal())                
            
    def run(self, batch_size, n_epochs, test_loader, per_epoch_testing = False):
        test_X, test_Y = test_loader
        
        ## TRAIN FOR `n_epochs` EPOCHS ##
        test_loader = (test_X, test_Y) if per_epoch_testing else None
        
        self.print_training_config(batch_size, n_epochs)
        self.train(batch_size, n_epochs, test_loader)
        
        if not per_epoch_testing:
            self.test(batch_size, test_X, test_Y)
