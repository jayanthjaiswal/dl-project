import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py

mat_names = glob.glob('./project_datasets/*.mat')
# each test subject got a different file

matfile = h5py.File(mat_names[2], 'r')

image_mat = matfile['image']
type_mat = matfile['type']

nans = np.sum(np.isnan(image_mat[:,:]))
type_set = set(type_mat[0,:])

# plt.plot(type_mat[0,:288]) # gets the significant values of types
# all the 0's occur after 288, and are meaningless I think
# so the image_mat, which has shape (288, 25, 1000) should correspond
# to the first 288 entries of type_mat, so
# for a single subject, training data should be image_mat, with 288 samples, each sample has shape (25, 1000)
# and our target label matrix should be type_mat[:288] (or 287?)
