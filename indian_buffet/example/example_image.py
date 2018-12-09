"""
Run IBP on the synthetic 'Cambridge Bars' dataset
"""
import sys
# import cPickle as CP

import numpy as np
from scipy.io import loadmat
from indian_buffet.main import IBP
from PIL import Image
from itertools import islice
import matplotlib.pyplot as plt
import matplotlib.cm as color_map

im = Image.open('lena_image.png').convert('LA')
im_array = np.array(im)[:, :, 0]


def break_patches(im, window_size=6):
    num_row, num_col = im.shape
    for x in range(0, num_row, window_size):
        if x + window_size <= num_row:
            for y in range(0, num_col, window_size):
                if y + window_size <= num_col:
                    yield (im[x:x+window_size, y:y+window_size]).flatten()


def reconstruct_from_patches(patches, num_cols=73):
    N, D = patches.shape
    window_size = int(np.round(np.sqrt(D)))
    patches_gen = (np.reshape(row, newshape=(window_size, window_size)) for row in data)
    # Two actions of logic in the next line
    #  - pick num_cols patches from the generator and concatenate over second axis
    #  - pick the resulting rows of [window_size x num_cols*window_size] and concatenate over first axis
    image = np.concatenate([np.concatenate(list(islice(patches_gen, num_cols)), axis=1) for _ in range(num_cols)],
                           axis=0)
    return image


data = np.stack(list(break_patches(im_array.astype(np.float32))))


# IBP parameter (gamma hyperparameters)
(alpha, alpha_a, alpha_b) = (1., 1., 1.)
# Observed data Gaussian noise (Gamma hyperparameters)
(sigma_x, sx_a, sx_b) = (1., 1., 1.)
# Latent feature weight Gaussian noise (Gamma hyperparameters)
(sigma_a, sa_a, sa_b) = (1., 1., 1.)

# Number of full sampling sweeps
numsamp = 12

# Center the data
(N,D) = data.shape
print(f'We have {N} samples of size {D}')

data_centered = IBP.centerData(data)

# Initialize the model
f = IBP(data_centered,(alpha, alpha_a, alpha_b),
        (sigma_x, sx_a, sx_b),
        (sigma_a, sa_a, sa_b), use_v=True)

# Do inference
for s in range(numsamp):
    # Print current chain state
    f.sample_report(s)

    # Take a new sample
    f.full_sample()
    print(f'Finished sample {s}')

if True:
    x_reconstruct = f.report_mean_x()
    im_reconstruct = reconstruct_from_patches(x_reconstruct, 73)
    Image.fromarray(im_reconstruct).show()


# Intensity plots of
# -ground truth factor-feature weights (top)
# -learned factor-feature weights (bottom)
K = max(0, len(f.weights()))  # len(trueWeights)
num_cols = int(np.ceil(np.sqrt(K)))

fig, axarr = plt.subplots(num_cols, num_cols)

for (idx, learnedFactor) in enumerate(f.weights()):
    x, y = int(idx / num_cols), idx % num_cols
    axarr[x, y].imshow(learnedFactor.reshape(6, 6), cmap=color_map.gray)

for n_row, axrow in enumerate(axarr):
    for n_col, ax in enumerate(axrow):
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])

        ax.tick_params(
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False)  # labels along the bottom edge are off
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)

plt.show()
