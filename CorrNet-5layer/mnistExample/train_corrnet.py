__author__ = 'Sarath'

import sys

sys.path.append("../Model/")
from corrnet import *

src_folder = sys.argv[1]+"matpic1/"
tgt_folder = sys.argv[2]

batch_size = 100
training_epochs = 50
l_rate = 0.01
optimization = "rmsprop"
tied = True
# n_visible_left = 392
# n_visible_right = 392
n_visible_left = 512
n_visible_right = 512
n_hidden_mid = 50
n_hidden = 200
lamda = 2
hidden_activation = "sigmoid"
output_activation = "sigmoid"
hidden_mid_activation = "sigmoid"
hidden_activation_prime = "sigmoid"
loss_fn = "squarrederror"

trainCorrNet(src_folder=src_folder, tgt_folder=tgt_folder, batch_size=batch_size,
             training_epochs=training_epochs, l_rate=l_rate, optimization=optimization,
             tied=tied, n_visible_left=n_visible_left, n_visible_right=n_visible_right,
             n_hidden=n_hidden, n_hidden_mid=n_hidden_mid, lamda=lamda, 
             hidden_activation=hidden_activation, output_activation=output_activation, 
             hidden_mid_activation=hidden_mid_activation, hidden_activation_prime=hidden_activation_prime, 
             loss_fn=loss_fn)

