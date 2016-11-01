__author__ = 'Sarath'

import sys

sys.path.append("../Model/")
from corrnet import *

src_folder = sys.argv[1]+"matpic1/"
tgt_folder = sys.argv[2]

batch_size = 100
training_epochs = 50
l_rate = 0.01
optimization = "nag"
tied = True
n_visible_left = 512
n_visible_right = 512
n_visible_left1 = 200
n_visible_right1 = 200
n_hidden_1024 = 1024
n_hidden_512 = 512
n_hidden_256 = 256
n_hidden_128 = 128
n_hidden_64 = 64
n_hidden_28 = 28
lamda = 2
hidden_activation = "relu"
output_activation = "relu"
hidden_activation_mid = "relu"
loss_fn = "squarrederror"

trainCorrNet(src_folder=src_folder, tgt_folder=tgt_folder, batch_size=batch_size,
             training_epochs=training_epochs, l_rate=l_rate, optimization=optimization,
             tied=tied, n_visible_left=n_visible_left, n_visible_right=n_visible_right,
             n_hidden=n_hidden, lamda=lamda, hidden_activation=hidden_activation,
             output_activation=output_activation, loss_fn=loss_fn)

