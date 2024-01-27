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

from discretegauss import sample_dgauss, scaled_noise_sample
import numpy as np

use_mux = False
CLIP = 0

class Optimizer:
    clip = CLIP
    iterations_done = 0
    noisy = False 

    def get_noise(self, sigma, n):
        noise = sfix.Array(n)
        @for_range_opt(n)
        def _(i):
            noise[i] = scaled_noise_sample(sigma)
        return noise.get_vector()

def log_e(x):
    return mpc_math.log_fx(x, math.e)

def set_n_threads(n_threads):
    Layer.n_threads = n_threads
    SGD.n_threads = n_threads
    
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
    clip = 4
    sigma = 4

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
    def __init__(self, N, d_in, d_out, d=1, activation='id', debug=False):
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
        self.W.alloc()
        self.b = sfix.Array(d_out)
        self.b.assign_all(0)

        back_N = N
        self.nabla_Y = Tensor([back_N, d, d_out], sfix)
        self.nabla_X = Tensor([back_N, d, d_in], sfix)
        self.nabla_W = sfix.Matrix(d_out, d_in)
        self.nabla_W.alloc()
        
        self.nabla_W_t = self.nabla_W.transpose()
        
        self.nabla_b = sfix.Array(d_out)
        
        self.nabla_W_per_sample_norms = sfix.Array(600)
        self.nabla_b_per_sample_norms = sfix.Array(600)
        self.nabla_b_clipped_sum = sfix.Array(self.d_out)
                
        if self.activation_layer:
            self.f_input = self.activation_layer.X
            self.activation_layer.Y = self.Y
            self.activation_layer.nabla_Y = self.nabla_Y
        else:
            self.f_input = self.Y
            
        self.clip = CLIP
            
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
        
        if self.clip:
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
   
    def get_L2_norm_sq(self, vec, sc=1):
        scale = sc
        vec = vec*(sfix(1/scale).expand_to_vector(len(vec)))
        sum_of_sq = sum(vec * vec)
        
        return sum_of_sq*scale*scale
   
    def get_L2_norm(self, vec, sc=1):
        scale = sc
        vec = vec*(sfix(1/scale).expand_to_vector(len(vec)))
        sum_of_sq = sum(vec * vec)
        L2_norm = mpc_math.sqrt(sum_of_sq)*(scale)
        return L2_norm
   
    def clip_vector(self, vec):
        scale = 1
        L2_norm = self.get_L2_norm(vec, scale)
        # print_ln("L2_norm = %s", L2_norm.reveal())
        mul = util.if_else(L2_norm > self.clip, (self.clip/L2_norm), sfix(1)).expand_to_vector(self.d_out)
        
        vec = vec * mul
        return vec*(sfix(scale).expand_to_vector(self.d_out))
   
    def accumulate_per_sample_norms(self, batch):
        N = len(batch)
        
        # FOR BIAS 
        assert N == self.nabla_b_per_sample_norms.total_size()
        @for_range_multithread(self.n_threads, 1, N)
        def _(i):
            nabla_Y_i = self.nabla_Y[i][0].get_vector()   
            L2_norm_sq = self.get_L2_norm_sq(nabla_Y_i)
            self.nabla_b_per_sample_norms[i] = (L2_norm_sq)
        
        # FOR WEIGHTS
        assert N == self.nabla_W_per_sample_norms.total_size()
        A = sfix.Matrix(N, self.d_out, address=self.nabla_Y.address)
        B = sfix.Matrix(self.N, self.d_in, address=self.X.address)

        #### THIS HAS BEEN MADE self.N IMMUNE
        offset = MemValue(self.optimizer.iterations_done*N) if not self.compute_nabla_X else MemValue(0)
        @for_range_multithread(self.n_threads, 1, N)
        def _(i):
            vec_c = B[i + offset].get_vector()
            sum_vec_c_2 = self.get_L2_norm_sq(vec_c)
            
            vec_r = A[i].get_vector()
            sum_vec_r_2 = self.get_L2_norm_sq(vec_r)

            L2_norm_sq = sum_vec_c_2 * sum_vec_r_2
            self.nabla_W_per_sample_norms[i] = (L2_norm_sq)
      
    def _clip_and_accumulate(self, batch, clipping_multiplier):
        N = len(batch)
        tmp = Matrix(self.d_in, self.d_out, unreduced_sfix)
        
        self.nabla_b.assign_vector(sum(sum(self.nabla_Y[k][j].get_vector() * (clipping_multiplier[k]).expand_to_vector(self.d_out)
                                           for k in range(N))
                                       for j in range(self.d)))
        
        # if self.optimizer.noisy:
        #     self.nabla_b.assign_vector(self.nabla_b.get_vector() +  self.optimizer.get_noise(16, self.d_out))

        ## WEIGHT CLIPPING ##
        A = sfix.Matrix(N, self.d_out, address=self.nabla_Y.address)
        B = sfix.Matrix(self.N, self.d_in, address=self.X.address)
        
        @for_range_opt_multithread(self.n_threads, N)
        def _(i):
            vec_r = A[i].get_vector()
            vec_r = vec_r * clipping_multiplier[i]
            A[i].assign_vector(vec_r)
            # A[i].__imul__(clipping_multiplier[i])

        ##### THIS HAS BEEN MADE self.N IMMUNE ##### 
        offset = MemValue(self.optimizer.iterations_done*N) if not self.compute_nabla_X else MemValue(0)
        @multithread(self.n_threads, self.d_in)
        def _(base, size):
            mp = B.direct_trans_mul(A, reduce = False,
                                        indices = (
                                            regint.inc(size, base),
                                            regint.inc(N, offset),
                                            regint.inc(N),
                                            regint.inc(self.d_out)
                                        )
                                    )
            tmp.assign_part_vector(mp, base)
            
        @multithread(self.n_threads, self.d_in * self.d_out,
                     max_size=get_program().budget)
        def _(base, size):
            # if self.optimizer.noisy:
            #     self.nabla_W_t.assign_vector(
            #         tmp.get_vector(base, size).reduce_after_mul() + self.optimizer.get_noise(16, size) , base=base)
            # else:
            self.nabla_W_t.assign_vector(
                tmp.get_vector(base, size).reduce_after_mul(), base=base)
        self.nabla_W.assign(self.nabla_W_t.transpose())
        
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
        
    def __forward(self, batch):
        N = len(batch)
        @for_range_opt(self.n_threads, N)
        def _(i):
            exp_x_i = exp(self.X[i])
            self.exp[batch[i]].assign_vector(exp_x_i)
            
            sum_exp_x_i = sum(exp_x_i) 
            self.sum_exp[batch[i]] = sum_exp_x_i
            
            softmax_vector = exp_x_i * (1/sum_exp_x_i).expand_to_vector(len(self.X[i]))
            cross_ent_softmax_loss = -sum(sfix.dot_product(self.Y[batch[i]], log_e(softmax_vector)))
            self.losses[batch[i]] = cross_ent_softmax_loss
       
    def _backward(self, batch):
        d_out = self.X.sizes[1]
        @for_range_opt_multithread(self.n_threads, len(batch))
        def _(i):
            div = softmax_from_exp(self.exp[i])
            self.nabla_X[i][:] = -self.Y[batch[i]][:] + div

        # print_ln("NABLA_X_0__ = %s", self.nabla_X[0][:100].reveal())        
            
    def __backward(self, batch):
        @for_range_opt(self.n_threads, len(batch))
        def _(i):
            div = self.exp[batch[i]].get_vector() * (1/self.sum_exp[batch[i]]).expand_to_vector(len(self.X[i]))
            self.nabla_X[i].assign_vector(-self.Y[batch[i]].get_vector() + div.get_vector())
            
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
    def __init__(self, layers: List[Layer], lr: float = 0.1):
        self.layers = layers
        self.learnable_layers = self.layers[:-1]
        self.nablas = []
        self.thetas = []
        self.per_sample_norms = []
        self.delta_thetas = []
        self.lr = MemValue(cfix(lr)) 
        self.epoch_num = MemValue(0)
        self.X_by_label = None
        self.shuffle = False
        self.momentum = 0
        self.early_division = False

        self.debug = True
        
        self.aggregated_norms = sfix.Array(600)
        
        prev_layer = None
        for i, layer in enumerate(self.layers):
            layer.optimizer = self
            if i != len(layers)-1:
                # all but the last layer
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
            # print_ln("%s", mpc_math.sqrt(p.get_vector()).reveal())
            self.aggregated_norms.assign_vector(self.aggregated_norms.get_vector() + p.get_vector())
            
        self.aggregated_norms.assign_vector(mpc_math.sqrt(self.aggregated_norms.get_vector()))
        # self.aggregated_norms.__iadd__(sfix(1e-6))
        # print_ln("AGG NORMS = %s", self.aggregated_norms.get_vector(0, 100).reveal())
        
        ### CODE FROM HERE        
        comparision: sfix = self.aggregated_norms.get_vector() > sfix(self.clip).expand_to_vector(N)
        # print_ln("COMPARISIONS = %s", comparision.reveal())
        
        self.eff_clipping_multiplier = sfix.Array(N)
        self.eff_clipping_multiplier.assign_vector(comparision.if_else(sfix(self.clip).expand_to_vector(N)/self.aggregated_norms.get_vector(), 1))
        # print_ln("EFF CLIP FACTORS = %s", self.eff_clipping_multiplier.get_vector(0, 100).reveal()) 
    
    def pre_update_clip_step(self, batch):
        assert self.clip and self.clip > 0
        
        self.calculate_eff_clip_multipliers(batch)
        for layer in reversed(self.learnable_layers):
            layer: Dense 
            layer._clip_and_accumulate(batch, self.eff_clipping_multiplier)
            
            # if self.noisy:
            #     for nabla in layer.nablas():
            #         # @multithread(self.n_threads, nabla.total_size(), max_size=get_program().budget)
            #         # def _(base, size):
            #         nabla.assign_vector(nabla.get_vector() + SGD.get_noise(16, nabla.total_size()))
    
    def update(self, i_epoch, i_batch, batch):        
        if self.clip:
            self.pre_update_clip_step(batch)
        
        for nabla, theta, delta_theta in zip(self.nablas, self.thetas, self.delta_thetas):
            if 0:
                if type(nabla) == Matrix:
                    print_ln("NABLA_W = %s", nabla[0].get_vector(0, min(nabla[0].total_size(), 1000)).reveal())
                else:
                    print_ln("NABLA_B = %s", nabla.get_vector(0, min(nabla.total_size(), 1000)).reveal())                                 

            @multithread(self.n_threads, nabla.total_size())
            def _(base, size):
                old = delta_theta.get_vector(base, size)
                red_old = self.momentum * old
                rate = self.lr.expand_to_vector(size)
                nabla_vector = nabla.get_vector(base, size)
                
                # if self.noisy:
                #     nabla_vector = nabla_vector + self.get_noise(16, size)
                
                # log_batch_size = math.log(len(batch), 2)
                # # divide by len(batch) by truncation
                # # increased rate if len(batch) is not a power of two
                
                # pre_trunc = nabla_vector.v * rate.v
                # k = max(nabla_vector.k, rate.k) + rate.f
                # m = rate.f + int(log_batch_size)
                # if self.early_division:
                #     v = pre_trunc
                # else:
                #     v = pre_trunc.round(k, m, signed=True,
                #                         nearest=sfix.round_nearest)
                
                # new = nabla_vector._new(v)
                new = nabla_vector / sfix(6000)
                diff = red_old - new
                delta_theta.assign_vector(diff, base)
                theta.assign_vector(theta.get_vector(base, size) +
                                    delta_theta.get_vector(base, size), base)
                
            if 1:
                if type(nabla) == Matrix:
                    print_ln("NABLA_W_SCALED = %s", delta_theta[0].get_vector(0, min(delta_theta[0].total_size(), 1000)).reveal())
                else:
                    print_ln("NABLA_B_SCALED = %s", delta_theta.get_vector(0, min(delta_theta.total_size(), 1000)).reveal())        

    def train(self, n_epochs, batch_size = 0):
        self.print_stats = True
        
        assert batch_size > 0
        
        N = batch_size
        self.n_correct = MemValue(0)
        
        @for_range(n_epochs)
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
            
            self.n_correct.write(0)
            train_loss = MemValue(sfix(0))
            self.iterations_done = MemValue(0)
            
            # TRAIN EACH BATCH
            @for_range(n_iterations)
            def _(j):
                print_str("Epoch %s.%s: ", i+1, j+1)
                batch = regint.Array(batch_size)
                    
                indices = indices_by_label[0]                
                batch.assign(indices.get_vector(j*batch_size, batch_size))
                
                self.forward(batch_size=batch_size, batch=batch)
                self.backward(batch=batch)
                self.update(i, j, batch)
                
                self.iterations_done.iadd(1)
                
                ### CALCULATE TRAIN STATS ###
                train_loss.iadd(self.layers[-1].l)
                batch_targets = self.layers[-1].Y.same_shape()
                batch_targets.assign_vector(self.layers[-1].Y.get_slice_vector(batch))
                self.n_correct.iadd(self.layers[-1].reveal_correctness(batch_size, batch_targets))

            if self.print_stats:
                print_ln("TRAIN LOSS = %s", (train_loss/n_iterations).reveal())
                print_ln("TRAIN ACC = (%s/%s)", self.n_correct.reveal(), self.layers[0].N)
                
    
    def run(batch_size):
        pass