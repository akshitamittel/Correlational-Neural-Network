__author__ = 'Sarath'

import time

from optimization import *
from Initializer import *
from NNUtil import *


import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class CorrNet(object):import pickle


    # 1. Get the number of neurons in a hidden layer
    def init(numpy_rng, theano_rng=None, l_rate=0.01, optimization="nag", tied=False, n_visible_left=None, n_visible_right=None, n_visible_left1=None, n_visible_right1=None, n_hidden_1024=None, n_hidden_512=None, n_hidden_256=None, n_hidden_128=None, n_hidden_64=None, n_hidden_28=None, lamda=5,W_left=None, W_right=None, W_left1=None, W_right1=None, W_left_prime=None, W_right_prime=None,W_left1_prime=None, W_right1_prime=None, W_mid_1024_512=None, W_mid_512_256=None, W_mid_256_128=None, W_mid_128_64=None, W_mid_64_28=None, W_mid_1024_512_prime=None,W_mid_512_256_prime=None, W_mid_256_128_prime=None, W_mid_128_64_prime=None, W_mid_64_28_prime=None,b_left=None, b_right=None, b_1024=None, b_512=None, b_256=None, b_128=None, b_64=None, b_28=None,b_1024_prime=None, b_512_prime=None, b_256_prime=None, b_128_prime=None, b_64_prime=None,b_prime_left1=None, b_prime_right1=None, b_prime_left=None, b_prime_right=None, input_left=None, input_right=None, hidden_activation="relu", hidden_activation_mid="relu", output_activation="relu", op_folder=None, loss_fn="squarrederror"):

        self.numpy_rng = numpy_rng
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        self.optimization = optimization
        self.l_rate = l_rate

        self.optimizer = get_optimizer(self.optimization, self.l_rate)
        self.Initializer = Initializer(self.numpy_rng)

        self.n_visible_left = n_visible_left
        self.n_visible_right = n_visible_right
        self.n_visible_left1 = n_visible_left1
        self.n_visible_right1 = n_visible_right1
        self.n_hidden_1024 = n_hidden_1024
        self.n_hidden_512 = n_hidden_512
        self.n_hidden_256 = n_hidden_256
        self.n_hidden_128 = n_hidden_128
        self.n_hidden_64 = n_hidden_64
        self.n_hidden_28 = n_hidden_28
        # 2. assign addtional hidden layers here
        self.lamda = lamda
        self.hidden_activation = hidden_activation
        self.hidden_activation_mid=hidden_activation_mid
        self.output_activation = output_activation
        self.loss_fn = loss_fn
        self.tied = tied
        self.op_folder = op_folder

        # 3. Add hidden to register variable
        self.W_left = self.Initializer.fan_based_sigmoid("W_left", W_left, n_visible_left, n_visible_left1)
        self.optimizer.register_variable("W_left",n_visible_left,n_visible_left1)

        self.W_right = self.Initializer.fan_based_sigmoid("W_right", W_right, n_visible_right, n_visible_right1)
        self.optimizer.register_variable("W_right",n_visible_right,n_visible_right1)

        self.W_right = self.Initializer.fan_based_sigmoid("W_left1", W_left1, n_visible_left1, n_hidden_1024)
        self.optimizer.register_variable("W_left1",n_visible_left1,n_hidden_1024)

        self.W_right = self.Initializer.fan_based_sigmoid("W_right1", W_right1, n_visible_right1, n_hidden_1024)
        self.optimizer.register_variable("W_right1",n_visible_right1,n_hidden_1024)

        self.W_mid_1024_512 = self.Initializer.fan_based_sigmoid("W_mid_1024_512", W_mid_1024_512, n_hidden_1024, n_hidden_512)
        self.optimizer.register_variable("W_mid_1024_512",n_hidden_1024 ,n_hidden_512)

        self.W_mid_512_256 = self.Initializer.fan_based_sigmoid("W_mid_512_256", W_mid_512_256, n_hidden_512, n_hidden_256)
        self.optimizer.register_variable("W_mid_512_256",n_hidden_512 ,n_hidden_256)

        self.W_mid_256_128 = self.Initializer.fan_based_sigmoid("W_mid_256_128", W_mid_256_128, n_hidden_256, n_hidden_128)
        self.optimizer.register_variable("W_mid_256_128",n_hidden_256 ,n_hidden_128)

        self.W_mid_128_64 = self.Initializer.fan_based_sigmoid("W_mid_128_64", W_mid_128_64, n_hidden_128, n_hidden_64)
        self.optimizer.register_variable("W_mid_128_64",n_hidden_128 ,n_hidden_64)

        self.W_mid_64_28 = self.Initializer.fan_based_sigmoid("W_mid_64_28", W_mid_64_28, n_hidden_64, n_hidden_28)
        self.optimizer.register_variable("W_mid_64_28",n_hidden_64 ,n_hidden_28)

        if not tied:
            self.W_left_prime = self.Initializer.fan_based_sigmoid("W_left_prime", W_left_prime, n_visible_left1, n_visible_left)
            self.optimizer.register_variable("W_left_prime",n_visible_left1, n_visible_left)
            self.W_right_prime = self.Initializer.fan_based_sigmoid("W_right_prime", W_right_prime, n_visible_right1, n_visible_right)
            self.optimizer.register_variable("W_right_prime",n_visible_right1, n_visible_right)
            self.W_left1_prime = self.Initializer.fan_based_sigmoid("W_left1_prime", W_left1_prime, n_hidden_1024, n_visible_left1)
            self.optimizer.register_variable("W_left1_prime",n_hidden_1024,n_visible_left1)
            self.W_right1_prime = self.Initializer.fan_based_sigmoid("W_right1_prime", W_right1_prime, n_hidden_1024, n_visible_right1)
            self.optimizer.register_variable("W_right1_prime",n_hidden_1024,n_visible_right1)
            self.W_mid_1024_512_prime = self.Initializer.fan_based_sigmoid("W_mid_1024_512_prime", W_mid_1024_512_prime, n_hidden_512, n_hidden_1024)
            self.optimizer.register_variable("W_mid_1024_512_prime",n_hidden_512,n_hidden_1024)
            self.W_mid_512_256_prime = self.Initializer.fan_based_sigmoid("W_mid_512_256_prime", W_mid_512_256_prime, n_hidden_256, n_hidden_512)
            self.optimizer.register_variable("W_mid_512_256_prime",n_hidden_256,n_hidden_512)
            self.W_mid_256_128_prime = self.Initializer.fan_based_sigmoid("W_mid_256_128_prime", W_mid_256_128_prime, n_hidden_128,n_hidden_256)
            self.optimizer.register_variable("W_mid_256_128_prime",n_hidden_128,n_hidden_256)
            self.W_mid_128_64_prime = self.Initializer.fan_based_sigmoid("W_mid_128_64_prime", W_mid_128_64_prime, n_hidden_64, n_hidden_128)
            self.optimizer.register_variable("W_mid_128_64_prime",n_hidden_64,n_hidden_128)
            self.W_mid_64_28_prime = self.Initializer.fan_based_sigmoid("W_mid_64_28_prime", W_mid_64_28_prime, n_hidden_28, n_hidden_64)
            self.optimizer.register_variable("W_mid_64_28_prime",n_hidden_28,n_hidden_64)
        else:
            self.W_left_prime = self.W_left.T
            self.W_right_prime = self.W_right.T
            self.W_left1_prime = self.W_left1.T
            self.W_right1_prime = self.W_right1.T
            self.W_mid_1024_512_prime = self.W_mid_1024_512.T
            self.W_mid_512_256_prime = self.W_mid_512_256.T
            self.W_mid_256_128_prime = self.W_mid_256_128.T
            self.W_mid_128_64_prime = self.W_mid_128_64.T
            self.W_mid_64_28_prime = self.W_mid_64_28.T

        """
        b_left=None, b_right=None, b_1024=None, b_512=None, b_256=None, b_128=None, b_64=None, b_28=None,
        b_1024_prime=None, b_512_prime=None, b_256_prime=None, b_128_prime=None, b_64_prime=None,
        b_prime_left1=None, b_prime_right1=None, b_prime_left=None, b_prime_right=None, 
        """
        #4. Add hidden to zero_vector
        self.b_left = self.Initializer.zero_vector("b_left", b_left, n_visible_left1)
        self.optimizer.register_variable("b_left",1,n_visible_left1)
        self.b_right = self.Initializer.zero_vector("b_right", b_right, n_visible_right1)
        self.optimizer.register_variable("b_right",1,n_visible_right1)

        self.b_1024 = self.Initializer.zero_vector("b_1024", b_1024, n_hidden_1024)
        self.optimizer.register_variable("b_1024",1,n_hidden_1024)
        self.b_512 = self.Initializer.zero_vector("b_512", b_512, n_hidden_512)
        self.optimizer.register_variable("b_512",1,n_hidden_512)
        self.b_256 = self.Initializer.zero_vector("b_256", b_256, n_hidden_256)
        self.optimizer.register_variable("b_256",1,n_hidden_256)
        self.b_128 = self.Initializer.zero_vector("b_128", b_128, n_hidden_128)
        self.optimizer.register_variable("b_128",1,n_hidden_128)
        self.b_64 = self.Initializer.zero_vector("b_64", b_64, n_hidden_64)
        self.optimizer.register_variable("b_64",1,n_hidden_64)
        self.b_28 = self.Initializer.zero_vector("b_28", b_28, n_hidden_28)
        self.optimizer.register_variable("b_28",1,n_hidden_28)
        self.b_1024_prime = self.Initializer.zero_vector("b_1024_prime", b_1024_prime, n_hidden_1024)
        self.optimizer.register_variable("b_1024_prime",1,n_hidden_1024)
        self.b_512_prime = self.Initializer.zero_vector("b_512_prime", b_512_prime, n_hidden_512)
        self.optimizer.register_variable("b_512_prime",1,n_hidden_512)
        self.b_256_prime = self.Initializer.zero_vector("b_256_prime", b_256_prime, n_hidden_256)
        self.optimizer.register_variable("b_256_prime",1,n_hidden_256)
        self.b_128_prime = self.Initializer.zero_vector("b_128_prime", b_128_prime, n_hidden_128)
        self.optimizer.register_variable("b_128_prime",1,n_hidden_128)
        self.b_64_prime = self.Initializer.zero_vector("b_64_prime", b_64_prime, n_hidden_64)
        self.optimizer.register_variable("b_64_prime",1,n_hidden_64)

        self.b_prime_left1 = self.Initializer.zero_vector("b_prime_left1", b_prime_left1, n_visible_left1)
        self.optimizer.register_variable("b_prime_left1",1,n_visible_left1)
        self.b_prime_right1 = self.Initializer.zero_vector("b_prime_right1", b_prime_right1, n_visible_right1)
        self.optimizer.register_variable("b_prime_right1",1,n_visible_right1)
        self.b_prime_left = self.Initializer.zero_vector("b_prime_left", b_prime_left, n_visible_left)
        self.optimizer.register_variable("b_prime_left",1,n_visible_left)
        self.b_prime_right = self.Initializer.zero_vector("b_prime_right", b_prime_right, n_visible_right)
        self.optimizer.register_variable("b_prime_right",1,n_visible_right)

        if input_left is None:
            self.x_left = T.matrix(name='x_left')
        else:
            self.x_left = input_left

        if input_right is None:
            self.x_right = T.matrix(name='x_right')
        else:
            self.x_right = input_right

        if tied:
            self.params = [self.W_left, self.W_right, self.b_left, self.b_right,
                 self.W_left1, self.W_right1, self.b_1024,
                 self.W_mid_1024_512, self.b_512, self.W_mid_512_256, self.b_256,
                 self.W_mid_256_128, self.b_128, self.W_mid_128_64, self.b_64,
                 self.W_mid_64_28, self.b_28, self.b_64_prime, self.b_128_prime,
                 self.b_256_prime, self.b_512_prime, self.b_1024_prime, 
                 self.b_prime_left1, self.b_prime_right1, self.b_prime_left, self.b_prime_right]
            self.param_names = ["W_left", "W_right", "b_left", "b_right",
                 "W_left1", "W_right1", "b_1024",
                 "W_mid_1024_512", "b_512", "W_mid_512_256", "b_256",
                 "W_mid_256_128", "b_128", "W_mid_128_64", "b_64",
                 "W_mid_64_28", "b_28", "b_64_prime", "b_128_prime",
                 "b_256_prime", "b_512_prime", "b_1024_prime", 
                 "b_prime_left1", "b_prime_right1", "b_prime_left", "b_prime_right"]
        else:
            self.params = [self.W_left, self.W_right, self.b_left, self.b_right,
                 self.W_left1, self.W_right1, self.b_1024,
                 self.W_mid_1024_512, self.b_512, self.W_mid_512_256, self.b_256,
                 self.W_mid_256_128, self.b_128, self.W_mid_128_64, self.b_64,
                 self.W_mid_64_28, self.b_28, self.b_64_prime, self.b_128_prime,
                 self.b_256_prime, self.b_512_prime, self.b_1024_prime, 
                 self.b_prime_left1, self.b_prime_right1, self.b_prime_left, self.b_prime_right,
                 self.W_mid_1024_512_prime, self.W_mid_512_256_prime, self.W_mid_256_128_prime,
                 self.W_mid_128_64_prime, self.W_mid_64_28_prime]
            self.param_names = ["W_left", "W_right", "b_left", "b_right",
                 "W_left1", "W_right1", "b_1024",
                 "W_mid_1024_512", "b_512", "W_mid_512_256", "b_256",
                 "W_mid_256_128", "b_128", "W_mid_128_64", "b_64",
                 "W_mid_64_28", "b_28", "b_64_prime", "b_128_prime",
                 "b_256_prime", "b_512_prime", "b_1024_prime", 
                 "b_prime_left1", "b_prime_right1", "b_prime_left", "b_prime_right",
                 "W_mid_1024_512_prime", "W_mid_512_256_prime", "W_mid_256_128_prime",
                 "W_mid_128_64_prime", "W_mid_64_28_prime"]

        self.proj_from_left = theano.function([self.x_left],self.project_from_left())
        self.proj_from_right = theano.function([self.x_right],self.project_from_right())
        self.recon_from_left = theano.function([self.x_left],self.reconstruct_from_left())
        self.recon_from_right = theano.function([self.x_right],self.reconstruct_from_right())

        self.save_params()


    def train_common(self,mtype="1111"):

        # 5. Add hidden to activation

        y1_1_pre = T.dot(self.x_left, self.W_left) + self.b_left
        y1_1 = activation(y1_1_pre, self.hidden_activation)
        y1_2_pre = T.dot(y1_1, self.W_left1) + self.b_1024
        y1_2 = activation(y1_2_pre, self.hidden_activation_mid)
        y1_3_pre = T.dot(y1_2, self.W_mid_1024_512) + self.b_512
        y1_3 = activation(y1_3_pre, self.hidden_activation_mid)
        y1_4_pre = T.dot(y1_3, self.W_mid_512_256) + self.b_256
        y1_4 = activation(y1_4_pre, self.hidden_activation_mid)
        y1_5_pre = T.dot(y1_4, self.W_mid_256_128) + self.b_128
        y1_5 = activation(y1_5_pre, self.hidden_activation_mid)
        y1_6_pre = T.dot(y1_5, self.W_mid_128_64) + self.b_64
        y1_6 = activation(y1_6_pre, self.hidden_activation_mid)
        y1_7_pre = T.dot(y1_6, self.W_mid_64_28) + self.b_28
        y1_7 = activation(y1_7_pre, self.hidden_activation_mid)
        y1_8_pre = T.dot(y1_7, self.W_mid_64_28_prime) + self.b_64_prime
        y1_8 = activation(y1_8_pre, self.hidden_activation_mid)
        y1_9_pre = T.dot(y1_8, self.W_mid_128_64_prime) + self.b_128_prime
        y1_9 = activation(y1_9_pre, self.hidden_activation_mid)
        y1_10_pre = T.dot(y1_9, self.W_mid_256_128_prime) + self.b_256_prime
        y1_10 = activation(y1_10_pre, self.hidden_activation_mid)
        y1_11_pre = T.dot(y1_10, self.W_mid_512_256_prime) + self.b_512_prime
        y1_11 = activation(y1_11_pre, self.hidden_activation_mid)
        y1_12_pre = T.dot(y1_11, self.W_mid_1024_512_prime) + self.b_1024_prime
        y1_12 = activation(y1_12_pre, self.hidden_activation_mid)
        z1_left1_pre = T.dot(y1_12, self.W_left1_prime) + self.b_prime_left1
        z1_left1 = activation(z1_left1_pre, self.hidden_activation)
        z1_right1_pre = T.dot(y1_12, self.W_right1_prime) + self.b_prime_right1
        z1_right1 = activation(z1_right1_pre, self.hidden_activation)
        z1_left_pre = T.dot(z1_left1, self.W_left_prime) + self.b_prime_left
        z1_right_pre = T.dot(z1_right1,self.W_right_prime) + self.b_prime_right
        z1_left = activation(z1_left_pre, self.output_activation)
        z1_right = activation(z1_right_pre, self.output_activation)
        L1 = loss(z1_left, self.x_left, self.loss_fn) + loss(z1_right, self.x_right, self.loss_fn)

        y2_1_pre = T.dot(self.x_right, self.W_right) + self.b_right
        y2_1 = activation(y1_1_pre, self.hidden_activation)
        y2_2_pre = T.dot(y1_1, self.W_right1) + self.b_1024
        y2_2 = activation(y1_2_pre, self.hidden_activation_mid)
        y2_3_pre = T.dot(y1_2, self.W_mid_1024_512) + self.b_512
        y2_3 = activation(y1_3_pre, self.hidden_activation_mid)
        y2_4_pre = T.dot(y1_3, self.W_mid_512_256) + self.b_256
        y2_4 = activation(y1_4_pre, self.hidden_activation_mid)
        y2_5_pre = T.dot(y1_4, self.W_mid_256_128) + self.b_128
        y2_5 = activation(y1_5_pre, self.hidden_activation_mid)
        y2_6_pre = T.dot(y1_5, self.W_mid_128_64) + self.b_64
        y2_6 = activation(y1_6_pre, self.hidden_activation_mid)
        y2_7_pre = T.dot(y1_6, self.W_mid_64_28) + self.b_28
        y2_7 = activation(y1_7_pre, self.hidden_activation_mid)
        y2_8_pre = T.dot(y1_7, self.W_mid_64_28_prime) + self.b_64_prime
        y2_8 = activation(y1_8_pre, self.hidden_activation_mid)
        y2_9_pre = T.dot(y1_8, self.W_mid_128_64_prime) + self.b_128_prime
        y2_9 = activation(y1_9_pre, self.hidden_activation_mid)
        y2_10_pre = T.dot(y1_9, self.W_mid_256_128_prime) + self.b_256_prime
        y2_10 = activation(y1_10_pre, self.hidden_activation_mid)
        y2_11_pre = T.dot(y1_10, self.W_mid_512_256_prime) + self.b_512_prime
        y2_11 = activation(y1_11_pre, self.hidden_activation_mid)
        y2_12_pre = T.dot(y1_11, self.W_mid_1024_512_prime) + self.b_1024_prime
        y2_12 = activation(y1_12_pre, self.hidden_activation_mid)
        z2_left1_pre = T.dot(y1_12, self.W_left1_prime) + self.b_prime_left1
        z2_left1 = activation(z1_left1_pre, self.hidden_activation)
        z2_right1_pre = T.dot(y1_12, self.W_right1_prime) + self.b_prime_right1
        z2_right1 = activation(z1_right1_pre, self.hidden_activation)
        z2_left_pre = T.dot(z1_left1, self.W_left_prime) + self.b_prime_left
        z2_right_pre = T.dot(z1_right1,self.W_right_prime) + self.b_prime_right
        z2_left = activation(z1_left_pre, self.output_activation)
        z2_right = activation(z1_right_pre, self.output_activation)
        L2 = loss(z2_left, self.x_left, self.loss_fn) + loss(z2_right, self.x_right, self.loss_fn)

        y3_left_pre = T.dot(self.x_left, self.W_left) + self.b_left
        y3_right_pre = T.dot(self.x_right, self.W_right) + self.b_right
        y3_left = activation(y3_left_pre, self.hidden_activation)
        y3_right = activation(y3_right_pre, self.hidden_activation)
        y3_2_pre = T.dot(y3_left, self.W_left1) + T.dot(y3_right, self.W_right1) + self.b_1024
        y3_2 = activation(y1_2_pre, self.hidden_activation_mid)
        y3_3_pre = T.dot(y1_2, self.W_mid_1024_512) + self.b_512
        y3_3 = activation(y1_3_pre, self.hidden_activation_mid)
        y3_4_pre = T.dot(y1_3, self.W_mid_512_256) + self.b_256
        y3_4 = activation(y1_4_pre, self.hidden_activation_mid)
        y3_5_pre = T.dot(y1_4, self.W_mid_256_128) + self.b_128
        y3_5 = activation(y1_5_pre, self.hidden_activation_mid)
        y3_6_pre = T.dot(y1_5, self.W_mid_128_64) + self.b_64
        y3_6 = activation(y1_6_pre, self.hidden_activation_mid)
        y3_7_pre = T.dot(y1_6, self.W_mid_64_28) + self.b_28
        y3_7 = activation(y1_7_pre, self.hidden_activation_mid)
        y3_8_pre = T.dot(y1_7, self.W_mid_64_28_prime) + self.b_64_prime
        y3_8 = activation(y1_8_pre, self.hidden_activation_mid)
        y3_9_pre = T.dot(y1_8, self.W_mid_128_64_prime) + self.b_128_prime
        y3_9 = activation(y1_9_pre, self.hidden_activation_mid)
        y3_10_pre = T.dot(y1_9, self.W_mid_256_128_prime) + self.b_256_prime
        y3_10 = activation(y1_10_pre, self.hidden_activation_mid)
        y3_11_pre = T.dot(y1_10, self.W_mid_512_256_prime) + self.b_512_prime
        y3_11 = activation(y1_11_pre, self.hidden_activation_mid)
        y3_12_pre = T.dot(y1_11, self.W_mid_1024_512_prime) + self.b_1024_prime
        y3_12 = activation(y1_12_pre, self.hidden_activation_mid)
        z3_left1_pre = T.dot(y1_12, self.W_left1_prime) + self.b_prime_left1
        z3_left1 = activation(z1_left1_pre, self.hidden_activation)
        z3_right1_pre = T.dot(y1_12, self.W_right1_prime) + self.b_prime_right1
        z3_right1 = activation(z1_right1_pre, self.hidden_activation)
        z3_left_pre = T.dot(z1_left1, self.W_left_prime) + self.b_prime_left
        z3_right_pre = T.dot(z1_right1,self.W_right_prime) + self.b_prime_right
        z3_left = activation(z1_left_pre, self.output_activation)
        z3_right = activation(z1_right_pre, self.output_activation)
        L3 = loss(z3_left, self.x_left, self.loss_fn) + loss(z3_right, self.x_right, self.loss_fn)

        #Come back:
        y1_mean = T.mean(y1_7, axis=0)
        y1_centered = y1_7 - y1_mean
        y2_mean = T.mean(y2_7, axis=0)
        y2_centered = y2_7 - y2_mean
        corr_nr = T.sum(y1_centered * y2_centered, axis=0)
        corr_dr1 = T.sqrt(T.sum(y1_centered * y1_centered, axis=0)+1e-8)
        corr_dr2 = T.sqrt(T.sum(y2_centered * y2_centered, axis=0)+1e-8)
        corr_dr = corr_dr1 * corr_dr2
        corr = corr_nr/corr_dr
        L4 = T.sum(corr) * self.lamda

        ly4_pre = T.dot(self.x_left, self.W_left) + self.b
        ly4 = activation(ly4_pre, self.hidden_activation)
        lz4_right_pre = T.dot(ly4,self.W_right_prime) + self.b_prime_right
        lz4_right = activation(lz4_right_pre, self.output_activation)
        ry4_pre = T.dot(self.x_right, self.W_right) + self.b
        ry4 = activation(ry4_pre, self.hidden_activation)
        rz4_left_pre = T.dot(ry4,self.W_left_prime) + self.b_prime_left
        rz4_left = activation(rz4_left_pre, self.output_activation)
        L5 = loss(lz4_right, self.x_right, self.loss_fn) + loss(rz4_left, self.x_left, self.loss_fn)

        if mtype=="1111":
            print "1111"
            L = L1 + L2 + L3 - L4
        elif mtype=="1110":
            print "1110"
            L = L1 + L2 + L3
        elif mtype=="1101":
            print "1101"
            L = L1 + L2 - L4
        elif mtype == "0011":
            print "0011"
            L = L3 - L4
        elif mtype == "1100":
            print "1100"
            L = L1 + L2
        elif mtype == "0010":
            print "0010"
            L = L3
        elif mtype == "euc":
            print "euc"
            L = L5
        elif mtype == "euc-cor":
            print "euc-cor"
            L = L5 - L4

        cost = T.mean(L)

        gradients = T.grad(cost, self.params)
        updates = []
        for p,g,n in zip(self.params, gradients, self.param_names):
            gr, upd = self.optimizer.get_grad_update(n,g)
            updates.append((p,p+gr))
            updates.extend(upd)

        return cost, updates

    def train_left(self):

        y1_1_pre = T.dot(self.x_left, self.W_left) + self.b_left
        y1_1 = activation(y1_1_pre, self.hidden_activation)
        y1_2_pre = T.dot(y1_1, self.W_left1) + self.b_1024
        y1_2 = activation(y1_2_pre, self.hidden_activation_mid)
        y1_3_pre = T.dot(y1_2, self.W_mid_1024_512) + self.b_512
        y1_3 = activation(y1_3_pre, self.hidden_activation_mid)
        y1_4_pre = T.dot(y1_3, self.W_mid_512_256) + self.b_256
        y1_4 = activation(y1_4_pre, self.hidden_activation_mid)
        y1_5_pre = T.dot(y1_4, self.W_mid_256_128) + self.b_128
        y1_5 = activation(y1_5_pre, self.hidden_activation_mid)
        y1_6_pre = T.dot(y1_5, self.W_mid_128_64) + self.b_64
        y1_6 = activation(y1_6_pre, self.hidden_activation_mid)
        y1_7_pre = T.dot(y1_6, self.W_mid_64_28) + self.b_28
        y1_7 = activation(y1_7_pre, self.hidden_activation_mid)
        y1_8_pre = T.dot(y1_7, self.W_mid_64_28_prime) + self.b_64_prime
        y1_8 = activation(y1_8_pre, self.hidden_activation_mid)
        y1_9_pre = T.dot(y1_8, self.W_mid_128_64_prime) + self.b_128_prime
        y1_9 = activation(y1_9_pre, self.hidden_activation_mid)
        y1_10_pre = T.dot(y1_9, self.W_mid_256_128_prime) + self.b_256_prime
        y1_10 = activation(y1_10_pre, self.hidden_activation_mid)
        y1_11_pre = T.dot(y1_10, self.W_mid_512_256_prime) + self.b_512_prime
        y1_11 = activation(y1_11_pre, self.hidden_activation_mid)
        y1_12_pre = T.dot(y1_11, self.W_mid_1024_512_prime) + self.b_1024_prime
        y1_12 = activation(y1_12_pre, self.hidden_activation_mid)
        z1_left1_pre = T.dot(y1_12, self.W_left1_prime) + self.b_prime_left1
        z1_left1 = activation(z1_left1_pre, self.hidden_activation)
        z1_left_pre = T.dot(z1_left1, self.W_left_prime) + self.b_prime_left
        z1_left = activation(z1_left_pre, self.output_activation)
        L = loss(z1_left, self.x_left, self.loss_fn)
        cost = T.mean(L)

        if self.tied:
            curr_params = [self.W_left, self.b_left, self.W_left1, self.b_1024,
                 self.W_mid_1024_512, self.b_512, self.W_mid_512_256, self.b_256,
                 self.W_mid_256_128, self.b_128, self.W_mid_128_64, self.b_64,
                 self.W_mid_64_28, self.b_28, self.b_64_prime, self.b_128_prime, 
                 self.b_256_prime, self.b_512_prime, self.b_1024_prime, 
                 self.b_prime_left1, self.b_prime_left]
            curr_param_names = ["W_left", "b_left", "W_left1", "b_1024",
                 "W_mid_1024_512", "b_512", "W_mid_512_256", "b_256",
                 "W_mid_256_128", "b_128", "W_mid_128_64", "b_64",
                 "W_mid_64_28", "b_28", "b_64_prime", "b_128_prime", 
                 "b_256_prime", "b_512_prime", "b_1024_prime", 
                 "b_prime_left1", "b_prime_left"]
        else:
            curr_params = [self.W_left, self.b_left, self.W_left1, self.b_1024,
                 self.W_mid_1024_512, self.b_512, self.W_mid_512_256, self.b_256,
                 self.W_mid_256_128, self.b_128, self.W_mid_128_64, self.b_64,
                 self.W_mid_64_28, self.b_28, self.b_64_prime, self.b_128_prime, 
                 self.b_256_prime, self.b_512_prime, self.b_1024_prime, 
                 self.b_prime_left1, self.b_prime_left, self.W_mid_64_28_prime,
                 self.W_mid_128_64_prime, self.W_mid_256_128_prime, self.W_mid_512_256_prime,
                 self.W_mid_1024_512_prime, self.W_left1_prime, self.W_left_prime]
            curr_param_names = ["W_left", "b_left", "W_left1", "b_1024",
                 "W_mid_1024_512", "b_512", "W_mid_512_256", "b_256",
                 "W_mid_256_128", "b_128", "W_mid_128_64", "b_64",
                 "W_mid_64_28", "b_28", "b_64_prime", "b_128_prime", 
                 "b_256_prime", "b_512_prime", "b_1024_prime", 
                 "b_prime_left1", "b_prime_left", "W_mid_64_28_prime",
                 "W_mid_128_64_prime", "W_mid_256_128_prime", "W_mid_512_256_prime",
                 "W_mid_1024_512_prime", "W_left1_prime", "W_left_prime"]

        gradients = T.grad(cost, curr_params)
        updates = []
        for p,g,n in zip(curr_params, gradients, curr_param_names):
            gr, upd = self.optimizer.get_grad_update(n,g)
            updates.append((p,p+gr))
            updates.extend(upd)
        return cost, updates

    def train_right(self):

        y2_1_pre = T.dot(self.x_right, self.W_right) + self.b_right
        y2_1 = activation(y1_1_pre, self.hidden_activation)
        y2_2_pre = T.dot(y1_1, self.W_right1) + self.b_1024
        y2_2 = activation(y1_2_pre, self.hidden_activation_mid)
        y2_3_pre = T.dot(y1_2, self.W_mid_1024_512) + self.b_512
        y2_3 = activation(y1_3_pre, self.hidden_activation_mid)
        y2_4_pre = T.dot(y1_3, self.W_mid_512_256) + self.b_256
        y2_4 = activation(y1_4_pre, self.hidden_activation_mid)
        y2_5_pre = T.dot(y1_4, self.W_mid_256_128) + self.b_128
        y2_5 = activation(y1_5_pre, self.hidden_activation_mid)
        y2_6_pre = T.dot(y1_5, self.W_mid_128_64) + self.b_64
        y2_6 = activation(y1_6_pre, self.hidden_activation_mid)
        y2_7_pre = T.dot(y1_6, self.W_mid_64_28) + self.b_28
        y2_7 = activation(y1_7_pre, self.hidden_activation_mid)
        y2_8_pre = T.dot(y1_7, self.W_mid_64_28_prime) + self.b_64_prime
        y2_8 = activation(y1_8_pre, self.hidden_activation_mid)
        y2_9_pre = T.dot(y1_8, self.W_mid_128_64_prime) + self.b_128_prime
        y2_9 = activation(y1_9_pre, self.hidden_activation_mid)
        y2_10_pre = T.dot(y1_9, self.W_mid_256_128_prime) + self.b_256_prime
        y2_10 = activation(y1_10_pre, self.hidden_activation_mid)
        y2_11_pre = T.dot(y1_10, self.W_mid_512_256_prime) + self.b_512_prime
        y2_11 = activation(y1_11_pre, self.hidden_activation_mid)
        y2_12_pre = T.dot(y1_11, self.W_mid_1024_512_prime) + self.b_1024_prime
        y2_12 = activation(y1_12_pre, self.hidden_activation_mid)
        z2_right1_pre = T.dot(y1_12, self.W_right1_prime) + self.b_prime_right1
        z2_right1 = activation(z1_right1_pre, self.hidden_activation)
        z2_right_pre = T.dot(z1_right1,self.W_right_prime) + self.b_prime_right
        z2_right = activation(z1_right_pre, self.output_activation)
        L = loss(z2_right, self.x_right, self.loss_fn)
        cost = T.mean(L)

        if self.tied:
            curr_params = [self.W_left, self.b_left, self.W_left1, self.b_1024,
                 self.W_mid_1024_512, self.b_512, self.W_mid_512_256, self.b_256,
                 self.W_mid_256_128, self.b_128, self.W_mid_128_64, self.b_64,
                 self.W_mid_64_28, self.b_28, self.b_64_prime, self.b_128_prime, 
                 self.b_256_prime, self.b_512_prime, self.b_1024_prime, 
                 self.b_prime_left1, self.b_prime_left]
            curr_param_names = ["W_left", "b_left", "W_left1", "b_1024",
                 "W_mid_1024_512", "b_512", "W_mid_512_256", "b_256",
                 "W_mid_256_128", "b_128", "W_mid_128_64", "b_64",
                 "W_mid_64_28", "b_28", "b_64_prime", "b_128_prime", 
                 "b_256_prime", "b_512_prime", "b_1024_prime", 
                 "b_prime_left1", "b_prime_left"]
        else:
            curr_params = [self.W_right, self.b_right, self.W_right1, self.b_1024,
                 self.W_mid_1024_512, self.b_512, self.W_mid_512_256, self.b_256,
                 self.W_mid_256_128, self.b_128, self.W_mid_128_64, self.b_64,
                 self.W_mid_64_28, self.b_28, self.b_64_prime, self.b_128_prime, 
                 self.b_256_prime, self.b_512_prime, self.b_1024_prime, 
                 self.b_prime_left1, self.b_prime_left, self.W_mid_64_28_prime,
                 self.W_mid_128_64_prime, self.W_mid_256_128_prime, self.W_mid_512_256_prime,
                 self.W_mid_1024_512_prime, self.W_right1_prime, self.W_right_prime]
            curr_param_names = ["W_right", "b_right", "W_right1", "b_1024",
                 "W_mid_1024_512", "b_512", "W_mid_512_256", "b_256",
                 "W_mid_256_128", "b_128", "W_mid_128_64", "b_64",
                 "W_mid_64_28", "b_28", "b_64_prime", "b_128_prime", 
                 "b_256_prime", "b_512_prime", "b_1024_prime", 
                 "b_prime_left1", "b_prime_left", "W_mid_64_28_prime",
                 "W_mid_128_64_prime", "W_mid_256_128_prime", "W_mid_512_256_prime",
                 "W_mid_1024_512_prime", "W_right1_prime", "W_right_prime"]

        gradients = T.grad(cost, curr_params)
        updates = []
        for p,g,n in zip(curr_params, gradients, curr_param_names):
            gr, upd = self.optimizer.get_grad_update(n,g)
            updates.append((p,p+gr))
            updates.extend(upd)
        return cost, updates

    def project_from_left(self):

        y1_1_pre = T.dot(self.x_left, self.W_left) + self.b_left
        y1_1 = activation(y1_1_pre, self.hidden_activation)
        y1_2_pre = T.dot(y1_1, self.W_left1) + self.b_1024
        y1_2 = activation(y1_2_pre, self.hidden_activation_mid)
        y1_3_pre = T.dot(y1_2, self.W_mid_1024_512) + self.b_512
        y1_3 = activation(y1_3_pre, self.hidden_activation_mid)
        y1_4_pre = T.dot(y1_3, self.W_mid_512_256) + self.b_256
        y1_4 = activation(y1_4_pre, self.hidden_activation_mid)
        y1_5_pre = T.dot(y1_4, self.W_mid_256_128) + self.b_128
        y1_5 = activation(y1_5_pre, self.hidden_activation_mid)
        y1_6_pre = T.dot(y1_5, self.W_mid_128_64) + self.b_64
        y1_6 = activation(y1_6_pre, self.hidden_activation_mid)
        y1_7_pre = T.dot(y1_6, self.W_mid_64_28) + self.b_28
        y1_7 = activation(y1_7_pre, self.hidden_activation_mid)
        return y1_7

    def project_from_right(self):

        y2_1_pre = T.dot(self.x_right, self.W_right) + self.b_right
        y2_1 = activation(y1_1_pre, self.hidden_activation)
        y2_2_pre = T.dot(y1_1, self.W_right1) + self.b_1024
        y2_2 = activation(y1_2_pre, self.hidden_activation_mid)
        y2_3_pre = T.dot(y1_2, self.W_mid_1024_512) + self.b_512
        y2_3 = activation(y1_3_pre, self.hidden_activation_mid)
        y2_4_pre = T.dot(y1_3, self.W_mid_512_256) + self.b_256
        y2_4 = activation(y1_4_pre, self.hidden_activation_mid)
        y2_5_pre = T.dot(y1_4, self.W_mid_256_128) + self.b_128
        y2_5 = activation(y1_5_pre, self.hidden_activation_mid)
        y2_6_pre = T.dot(y1_5, self.W_mid_128_64) + self.b_64
        y2_6 = activation(y1_6_pre, self.hidden_activation_mid)
        y2_7_pre = T.dot(y1_6, self.W_mid_64_28) + self.b_28
        y2_7 = activation(y1_7_pre, self.hidden_activation_mid)
        return y2_7

    def reconstruct_from_left(self):

        y1_1_pre = T.dot(self.x_left, self.W_left) + self.b_left
        y1_1 = activation(y1_1_pre, self.hidden_activation)
        y1_2_pre = T.dot(y1_1, self.W_left1) + self.b_1024
        y1_2 = activation(y1_2_pre, self.hidden_activation_mid)
        y1_3_pre = T.dot(y1_2, self.W_mid_1024_512) + self.b_512
        y1_3 = activation(y1_3_pre, self.hidden_activation_mid)
        y1_4_pre = T.dot(y1_3, self.W_mid_512_256) + self.b_256
        y1_4 = activation(y1_4_pre, self.hidden_activation_mid)
        y1_5_pre = T.dot(y1_4, self.W_mid_256_128) + self.b_128
        y1_5 = activation(y1_5_pre, self.hidden_activation_mid)
        y1_6_pre = T.dot(y1_5, self.W_mid_128_64) + self.b_64
        y1_6 = activation(y1_6_pre, self.hidden_activation_mid)
        y1_7_pre = T.dot(y1_6, self.W_mid_64_28) + self.b_28
        y1_7 = activation(y1_7_pre, self.hidden_activation_mid)
        y1_8_pre = T.dot(y1_7, self.W_mid_64_28_prime) + self.b_64_prime
        y1_8 = activation(y1_8_pre, self.hidden_activation_mid)
        y1_9_pre = T.dot(y1_8, self.W_mid_128_64_prime) + self.b_128_prime
        y1_9 = activation(y1_9_pre, self.hidden_activation_mid)
        y1_10_pre = T.dot(y1_9, self.W_mid_256_128_prime) + self.b_256_prime
        y1_10 = activation(y1_10_pre, self.hidden_activation_mid)
        y1_11_pre = T.dot(y1_10, self.W_mid_512_256_prime) + self.b_512_prime
        y1_11 = activation(y1_11_pre, self.hidden_activation_mid)
        y1_12_pre = T.dot(y1_11, self.W_mid_1024_512_prime) + self.b_1024_prime
        y1_12 = activation(y1_12_pre, self.hidden_activation_mid)
        z1_left1_pre = T.dot(y1_12, self.W_left1_prime) + self.b_prime_left1
        z1_left1 = activation(z1_left1_pre, self.hidden_activation)
        z1_right1_pre = T.dot(y1_12, self.W_right1_prime) + self.b_prime_right1
        z1_right1 = activation(z1_right1_pre, self.hidden_activation)
        z1_left_pre = T.dot(z1_left1, self.W_left_prime) + self.b_prime_left
        z1_right_pre = T.dot(z1_right1,self.W_right_prime) + self.b_prime_right
        z1_left = activation(z1_left_pre, self.output_activation)
        z1_right = activation(z1_right_pre, self.output_activation)
        return z1_left, z1_right

    def reconstruct_from_right(self):

        y2_1_pre = T.dot(self.x_right, self.W_right) + self.b_right
        y2_1 = activation(y1_1_pre, self.hidden_activation)
        y2_2_pre = T.dot(y1_1, self.W_right1) + self.b_1024
        y2_2 = activation(y1_2_pre, self.hidden_activation_mid)
        y2_3_pre = T.dot(y1_2, self.W_mid_1024_512) + self.b_512
        y2_3 = activation(y1_3_pre, self.hidden_activation_mid)
        y2_4_pre = T.dot(y1_3, self.W_mid_512_256) + self.b_256
        y2_4 = activation(y1_4_pre, self.hidden_activation_mid)
        y2_5_pre = T.dot(y1_4, self.W_mid_256_128) + self.b_128
        y2_5 = activation(y1_5_pre, self.hidden_activation_mid)
        y2_6_pre = T.dot(y1_5, self.W_mid_128_64) + self.b_64
        y2_6 = activation(y1_6_pre, self.hidden_activation_mid)
        y2_7_pre = T.dot(y1_6, self.W_mid_64_28) + self.b_28
        y2_7 = activation(y1_7_pre, self.hidden_activation_mid)
        y2_8_pre = T.dot(y1_7, self.W_mid_64_28_prime) + self.b_64_prime
        y2_8 = activation(y1_8_pre, self.hidden_activation_mid)
        y2_9_pre = T.dot(y1_8, self.W_mid_128_64_prime) + self.b_128_prime
        y2_9 = activation(y1_9_pre, self.hidden_activation_mid)
        y2_10_pre = T.dot(y1_9, self.W_mid_256_128_prime) + self.b_256_prime
        y2_10 = activation(y1_10_pre, self.hidden_activation_mid)
        y2_11_pre = T.dot(y1_10, self.W_mid_512_256_prime) + self.b_512_prime
        y2_11 = activation(y1_11_pre, self.hidden_activation_mid)
        y2_12_pre = T.dot(y1_11, self.W_mid_1024_512_prime) + self.b_1024_prime
        y2_12 = activation(y1_12_pre, self.hidden_activation_mid)
        z2_left1_pre = T.dot(y1_12, self.W_left1_prime) + self.b_prime_left1
        z2_left1 = activation(z1_left1_pre, self.hidden_activation)
        z2_right1_pre = T.dot(y1_12, self.W_right1_prime) + self.b_prime_right1
        z2_right1 = activation(z1_right1_pre, self.hidden_activation)
        z2_left_pre = T.dot(z1_left1, self.W_left_prime) + self.b_prime_left
        z2_right_pre = T.dot(z1_right1,self.W_right_prime) + self.b_prime_right
        z2_left = activation(z1_left_pre, self.output_activation)
        z2_right = activation(z1_right_pre, self.output_activation)
        return z2_left, z2_right

    def get_lr_rate(self):
        return self.optimizer.get_l_rate()

    def set_lr_rate(self,new_lr):
        self.optimizer.set_l_rate(new_lr)

    def save_matrices(self):

        for p,nm in zip(self.params, self.param_names):
            numpy.save(self.op_folder+nm, p.get_value(borrow=True))

    def save_params(self):

        params = {}
        params["optimization"] = self.optimization
        params["l_rate"] = self.l_rate
        params["n_visible_left"] = self.n_visible_left
        params["n_visible_right"] = self.n_visible_right
        params["n_visible_left1"] = self.n_visible_left1
        params["n_visible_right1"] = self.n_visible_right1
        params["n_hidden_1024"] = self.n_hidden_1024
        params["n_hidden_512"] = self.n_hidden_512
        params["n_hidden_256"] = self.n_hidden_256
        params["n_hidden_128"] = self.n_hidden_128
        params["n_hidden_64"] = self.n_hidden_64
        params["n_hidden_28"] = self.n_hidden_28
        #save new hidden parameters
        params["lamda"] = self.lamda
        params["hidden_activation"] = self.hidden_activation
        params["hidden_activation_mid"] = self.hidden_activation_mid
        params["output_activation"] = self.output_activation
        params["loss_fn"] = self.loss_fn
        params["tied"] = self.tied
        params["numpy_rng"] = self.numpy_rng
        params["theano_rng"] = self.theano_rng

        pickle.dump(params,open(self.op_folder+"params.pck","wb"),-1)


    def load(self, folder, input_left=None, input_right=None):

        plist = pickle.load(open(folder+"params.pck","rb"))

        # 6. Add hidden params
        self.init(plist["numpy_rng"], theano_rng=plist["theano_rng"], l_rate=plist["l_rate"],
                  optimization=plist["optimization"], tied=plist["tied"],
                  n_visible_left=plist["n_visible_left"], n_visible_right=plist["n_visible_right"],
                  n_hidden=plist["n_hidden"], lamda=plist["lamda"], W_left=folder+"W_left",
                  W_right=folder+"W_right", b=folder+"b", W_left_prime=folder+"W_left_prime",
                  W_right_prime=folder+"W_right_prime", b_prime_left=folder+"b_prime_left",
                  b_prime_right=folder+"b_prime_right", input_left=input_left, input_right=input_right,
                  hidden_activation=plist["hidden_activation"], output_activation=plist["output_activation"],
                  loss_fn = plist["loss_fn"], op_folder=folder)



# 7. Add hidden params
def trainCorrNet(src_folder, tgt_folder, batch_size = 20, training_epochs=40,
                 l_rate=0.01, optimization="nag", tied=False, n_visible_left=None,
                 n_visible_right=None, n_visible_left1=None, n_visible_right1=None,
                 n_hidden_1024=None, n_hidden_512=None, n_hidden_256=None,
                 n_hidden_128=None, n_hidden_64=None, n_hidden_28=None, lamda=5,
                 W_left=None, W_right=None, b_left=None, b_right=None, W_left1=None,
                 W_right1=None, b_1024=None, W_mid_1024_512=None, b_512=None,
                 W_mid_512_256=None, b_256=None, W_mid_256_128=None, b_128=None,
                 W_mid_128_64=None, b_64=None, W_mid_64_28=None, b_28=None,
                 W_mid_64_28_prime=None, b_64_prime=None, W_mid_128_64_prime=None, b_128_prime=None,
                 W_mid_256_128_prime=None, b_256_prime=None, W_mid_512_256_prime=None, b_512_prime=None,
                 W_mid_1024_512_prime=None, b_1024_prime=None, W_left1_prime=None, b_prime_left1=None,
                 W_left_prime=None, b_prime_left=None, W_right1_prime=None, b_prime_right1=None,
                 W_right_prime=None, b_prime_right=None,hidden_activation="relu", hidden_activation_mid="relu",
                 output_activation="relu", loss_fn = "squarrederror"):

    index = T.lscalar()
    x_left = T.matrix('x_left')
    x_right = T.matrix('x_right')

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # 8. Add hidden params
    model = CorrNet()
    model.init(numpy_rng=rng, theano_rng=theano_rng, l_rate=l_rate, optimization=optimization, tied=tied, n_visible_left=n_visible_left, n_visible_right=n_visible_right, n_hidden=n_hidden, lamda=lamda, W_left=W_left, W_right=W_right, b=b, W_left_prime=W_left_prime, W_right_prime=W_right_prime, b_prime_left=b_prime_left, b_prime_right=b_prime_right, input_left=x_left, input_right=x_right, hidden_activation=hidden_activation, output_activation=output_activation, loss_fn =loss_fn, op_folder=tgt_folder)
    #model.load(tgt_folder,x_left,x_right)
    start_time = time.clock()
    train_set_x_left = theano.shared(numpy.asarray(numpy.zeros((1000,n_visible_left)), dtype=theano.config.floatX), borrow=True)
    train_set_x_right = theano.shared(numpy.asarray(numpy.zeros((1000,n_visible_right)), dtype=theano.config.floatX), borrow=True)

    common_cost, common_updates = model.train_common("1111")
    mtrain_common = theano.function([index], common_cost,updates=common_updates,givens=[(x_left, train_set_x_left[index * batch_size:(index + 1) * batch_size]),(x_right, train_set_x_right[index * batch_size:(index + 1) * batch_size])])

    left_cost, left_updates = model.train_left()
    mtrain_left = theano.function([index], left_cost,updates=left_updates,givens=[(x_left, train_set_x_left[index * batch_size:(index + 1) * batch_size])])

    right_cost, right_updates = model.train_right()
    mtrain_right = theano.function([index], right_cost,updates=right_updates,givens=[(x_right, train_set_x_right[index * batch_size:(index + 1) * batch_size])])


    diff = 0
    flag = 1
    detfile = open(tgt_folder+"details.txt","w")
    detfile.close()
    oldtc = float("inf")

    for epoch in xrange(training_epochs):

        print "in epoch ", epoch
        c = []
        ipfile = open(src_folder+"train/ip.txt","r")
        for line in ipfile:
            next = line.strip().split(",")
            if(next[0]=="xy"):
                if(next[1]=="dense"):
                    denseTheanoloader(next[2]+"_left",train_set_x_left,"float32")
                    denseTheanoloader(next[2]+"_right",train_set_x_right, "float32")
                else:
                    sparseTheanoloader(next[2]+"_left",train_set_x_left,"float32",1000,n_visible_left)
                    sparseTheanoloader(next[2]+"_right",train_set_x_right, "float32", 1000, n_visible_right)
                for batch_index in range(0,int(next[3])/batch_size):
                    c.append(mtrain_common(batch_index))
            elif(next[0]=="x"):
                if(next[1]=="dense"):
                    denseTheanoloader(next[2]+"_left",train_set_x_left,"float32")
                else:
                    sparseTheanoloader(next[2]+"_left",train_set_x_left,"float32",1000,n_visible_left)
                for batch_index in range(0,int(next[3])/batch_size):
                    c.append(mtrain_left(batch_index))
            elif(next[0]=="y"):
                if(next[1]=="dense"):
                    denseTheanoloader(next[2]+"_right",train_set_x_right,"float32")
                else:
                    sparseTheanoloader(next[2]+"_right",train_set_x_right,"float32",1000,n_visible_right)
                for batch_index in range(0,int(next[3])/batch_size):
                    c.append(mtrain_right(batch_index))


        if(flag==1):
            flag = 0
            diff = numpy.mean(c)
            di = diff
        else:
            di = numpy.mean(c) - diff
            diff = numpy.mean(c)

        print 'Difference between 2 epochs is ', di
        print 'Training epoch %d, cost ' % epoch, diff

        ipfile.close()

        detfile = open(tgt_folder+"details.txt","a")
        detfile.write("train\t"+str(diff)+"\n")
        detfile.close()
        # save the parameters for every 5 epochs
        if((epoch+1)%5==0):
            model.save_matrices()

    end_time = time.clock()
    training_time = (end_time - start_time)
    print ' code ran for %.2fm' % (training_time / 60.)
    model.save_matrices()
