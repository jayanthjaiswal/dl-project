import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py

mat_names = glob.glob('./project_datasets/*.mat')
# each test subject got a different file

# plt.plot(type_mat[0,:288]) # gets the significant values of types
# all the 0's occur after 288, and are meaningless I think
# so the image_mat, which has shape (288, 25, 1000) should correspond
# to the first 288 entries of type_mat, so
# for a single subject, training data should be image_mat, with 288 samples, each sample has shape (25, 1000)
# and our target label matrix should be type_mat[:288] (or 287?)

# all of the data:

all_files = [h5py.File(m, 'r') for m in mat_names]
all_ims = [f['image'] for f in all_files]
all_types = [f['type'] for f in all_files]
