#!/usr/bin/env python

"""
Methods to perform Hadamard transforms using different
orderings:

1. natural
2. dyadic/Paley
3. Sequency

Still not dure which to use when
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


############################################
# CS experiment simulation based on Zibang #
############################################
def run_exp(target, n_pixel, n_coef, n_step, path_matrix, order='nat'):
    int_map = np.zeros((n_pixel, n_pixel, n_step))
    a_mat = np.zeros((n_coef, n_pixel**2))
    for i in range(n_coef):
        if not i % 50:
            print(i, end='\r')
        a = np.argwhere(path_matrix == i)
        row, col = a[0][0], a[0][1]

        for step in range(n_step):
            pattern = get_had_pat(n_pixel, row, col, (-1)**step, order=order)
            a_mat[i, :] = pattern.flatten()
            int_map[row, col, step] = np.sum(pattern * target)
    if n_step == 1:
        int_map = np.squeeze(int_map)
    return int_map, a_mat


def get_had_pat(n_point, u, v, init_phase, order='nat'):
    spec = np.zeros((n_point, n_point))
    # delta func
    spec[u, v] = 1
    if order in ['nat', 'dya']:
        had_pat = (fwht2_wiki(spec, inv=True, order=order) + 1) / 2
    else:  # meaning sequency
        had_pat = (fwht2_seq(spec, inv=True) + 1) / 2

    if init_phase == -1:
        had_pat = 1 - had_pat

    return had_pat


def hspi_recon(data, n_step, order='nat'):
    if n_step == 2:
        spec = data[:, :, 0] - data[:, :, 1]
    elif n_step == 1:
        spec = data.copy()
    else:
        raise NotImplementedError

    if order in ['nat', 'dya']:
        img = fwht2_wiki(spec/spec.size, inv=True, order=order)
    else:  # sequency
        img = fwht2_seq(spec/spec.size, inv=True)
    return img, spec


def hspi_subset_recon(data, n_coefs, n_step, path_matrix, order='nat'):
    if n_step == 2:
        spec = data[:, :, 0] - data[:, :, 1]
    elif n_step == 1:
        spec = data.copy()

    # here I could use some np.where filtering based on 
    sub_spec = np.zeros(spec.shape)
    for i in range(n_coefs):
        a = np.argwhere(path_matrix == i)
        row, col = a[0][0], a[0][1]
        sub_spec[row, col] = spec[row, col]

    if order in ['nat', 'dya']:
        img = fwht2_wiki(sub_spec/sub_spec.size, inv=True, order=order)
    else:  # sequency
        img = fwht2_seq(sub_spec/sub_spec.size, inv=True)
    return img, sub_spec


def psnr(target, img):
    """
    Adapted from Ducron, SPIRiT
    """
    if target.shape != img.shape:
        raise IndexError

    mx_target = np.amax(target)
    diff = target - img
    MSE = 1/diff.size * np.sum(diff**2)

    return (10 * np.log10(mx_target**2 / MSE))


#####################################
# 2D transforms based on 1D methods #
#####################################
def fwht2_wiki(arr, inv=False, order='nat'):
    ans = np.zeros(arr.shape)
    for i in range(len(arr)):
        ans[i, :] = fwht_wiki(arr[i, :], inv=inv, order=order)

    for j in range(len(arr)):
        ans[:, j] = fwht_wiki(ans[:, j], inv, order)
    # print('inverse',inv)
    return ans


def fwht2_seq(arr, inv=False):
    ans = np.zeros(arr.shape)
    for i in range(len(arr)):
        ans[i, :] = fwht_seq(arr[i, :], inv=inv)

    for j in range(len(arr)):
        ans[:, j] = fwht_seq(ans[:, j], inv=inv)
    return ans


##############
# 1D methods #
##############
def fwht_seq(data, inv=False):
    """
    adapted from zibang zhang 2015 paper
    """
    xx = bit_rev_order(data)
    x = np.asarray(xx.copy(), dtype=float)
    N = len(x)
    k1, k2, k3 = N, 1, N//2
    for i1 in range(1, int(np.log2(N))+1):
        L1 = 1
        for i2 in range(1, k2+1):
            for i3 in range(1, k3+1):
                i = i3 + L1 - 1
                j = i + k3
                temp1 = x[i-1]
                temp2 = x[j-1]
                if i2 % 2 == 0:
                    x[i-1] = temp1 - temp2
                    x[j-1] = temp1 + temp2
                else:
                    x[i-1] = temp1 + temp2
                    x[j-1] = temp1 - temp2

            L1 = L1 + k1
        k1, k2, k3 = k1//2, k2*2, k3//2
    if not inv:
        x = x/N
    return x


def fwht_wiki(a, inv=False, order='nat'):
    """Fast Walsh–Hadamard Transform of array a."""
    if order == 'dya':  # dyadic/Paley ordering
        b = bit_rev_order(a)
        ans = np.asarray(b.copy(), dtype=float)
    elif order == 'nat':
        ans = np.asarray(a.copy(), dtype=float)
    else:
        raise ValueError
    h = 1
    n = len(a)
    while 2*h <= n:
        # perform FWHT
        for i in range(0, n, 2*h):
            for j in range(i, i + h):
                x = ans[j]
                y = ans[j + h]
                ans[j] = x + y
                ans[j + h] = x - y
        h *= 2
    if not inv:
        ans = ans/n
    return ans


###########
# helpers #
###########
def zigzag(N):
    """
    Function to generate a square matrix of size NxN
    whose elements are integers going from 1 to N^2
    following the zig zag scan pattern
    used in image compression patterns, like JPEG.
    """
    mat = np.zeros((N, N), dtype=int)
    for n in range(1, N+1):
        mat[n-1, 0] = 1/2*(-1)**(n+1) * (n+1 + (-1)**(n+1) * ((n-1)*(n+1)+2)-2)

    for i in range(N):
        for j in range(1, N):
            if i % 2 == j % 2:
                mat[i, j] = mat[i, j-1] + 2*i + 1
            else:
                mat[i, j] = mat[i, j-1] + 2*j
    mat = np.fliplr(mat)

    for i in range(1, N):
        for j in range(N-1):
            if i > j:
                mat[i, j] = mat[i, j] - (i-j)**2

    return np.fliplr(mat)


def bit_rev_order(arr):
    f, e = math.frexp(len(arr))
    r = list(range( int(0.5*np.power(2, e)))) 
    binary = [bin(k)[2:].zfill(e-1) for k in r]
    return arr[[int(k[::-1], 2) for k in binary]]


if __name__ == "__main__":
    n_pixel = 128
    target = cv2.imread("C:\\Users\\David Palecek\\Documents\\UAlg\\my_opt\\Data\\Scripts\\hadamard\\lena.bmp")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    target = cv2.resize(target, (n_pixel, n_pixel), interpolation=cv2.INTER_AREA)

    n_coef = 2000  # n_pixel**2 # for full recon
    n_step = 2  # 2 if measurement balanced, 1, if not
    order = 'seq'  # options nat/dya/seq
    path = zigzag(n_pixel) - 1   # Preferentially trying to get high information content
    meas_result = run_exp(target, n_pixel, n_coef, n_step, path_matrix=path, order=order)

    img, spec = hspi_recon(meas_result, n_step, order=order)

    psn = psnr(target, img)
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(15, 6))
    ax[0].imshow(target)
    ax[0].set_title(np.amax(target))

    ax[1].imshow(img)
    ax[1].set_title(np.amax(img))

    ax[2].imshow(np.log(abs(spec)))
    ax[2].set_title(np.amax(spec))
    fig.suptitle('order: ' + order + ', PSNR: ' + str(np.round(psn, 2)) + ', comp ratio: ' + str(np.round(n_coef/target.size, 2)))
    plt.show()
