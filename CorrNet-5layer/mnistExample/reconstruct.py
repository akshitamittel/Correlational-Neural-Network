__author__ = 'Sarath'

import numpy
import os
import sys

sys.path.append("../Model/")
from corrnet import *

def create_folder(folder):

	if not os.path.exists(folder):
		os.makedirs(folder)



src_folder = sys.argv[1]+"matpic/"
tgt_folder = sys.argv[2]

model = CorrNet()
model.load(tgt_folder)

create_folder(tgt_folder+"reconstruct/")

mat = numpy.load(src_folder+"train/view1.npy")
matLeft, matRight = model.recon_from_left(mat)
numpy.save(tgt_folder+"reconstruct/train-view1_L",matLeft)
numpy.save(tgt_folder+"reconstruct/train-view1_R",matRight)

mat = numpy.load(src_folder+"train/view2.npy")
matLeft, matRight = model.recon_from_right(mat)
numpy.save(tgt_folder+"reconstruct/train-view2_L",matLeft)
numpy.save(tgt_folder+"reconstruct/train-view2_R",matRight)

mat = numpy.load(src_folder+"train/labels.npy")
numpy.save(tgt_folder+"reconstruct/train-labels",mat)


mat = numpy.load(src_folder+"valid/view1.npy")
matLeft, matRight = model.recon_from_left(mat)
numpy.save(tgt_folder+"reconstruct/valid-view1_L",matLeft)
numpy.save(tgt_folder+"reconstruct/valid-view1_R",matRight)

mat = numpy.load(src_folder+"valid/view2.npy")
matLeft, matRight = model.recon_from_right(mat)
numpy.save(tgt_folder+"reconstruct/valid-view2_L",matLeft)
numpy.save(tgt_folder+"reconstruct/valid-view2_L",matRight)

mat = numpy.load(src_folder+"valid/labels.npy")
numpy.save(tgt_folder+"reconstruct/valid-labels",mat)


mat = numpy.load(src_folder+"test/view1.npy")
matLeft, matRight = model.recon_from_left(mat)
numpy.save(tgt_folder+"reconstruct/test-view1_L",matLeft)
numpy.save(tgt_folder+"reconstruct/test-view1_R",matRight)

mat = numpy.load(src_folder+"test/view2.npy")
matLeft, matRight = model.recon_from_right(mat)
numpy.save(tgt_folder+"reconstruct/test-view2_L",matLeft)
numpy.save(tgt_folder+"reconstruct/test-view2_R",matRight)

mat = numpy.load(src_folder+"test/labels.npy")
numpy.save(tgt_folder+"reconstruct/test-labels",mat)

