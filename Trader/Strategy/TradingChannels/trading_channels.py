import itertools
import numpy as np
import math


class Permutation:
    def __init__(self,
                 time_series: np.ndarray,
                 embedding_order: int = 3,
                 time_delay: int = 1
                 ):
        self.x = time_series
        self.order = embedding_order
        self.delay = time_delay
        self.embed = self._gen_embed()
        self.all_perms = list(itertools.permutations(np.arange(0, embedding_order), embedding_order))
        self.frqs = self._gen_frq()

    def _gen_embed(self):

        if self.x.ndim == 1:
            # pass 1D array

            y = np.zeros((self.order, self.x.shape[-1] - (self.order - 1) * self.delay))
            for i in range(self.order):
                y[i] = self.x[(i * self.delay):(i * self.delay + y.shape[1])]
            return y.T

        else:

            y = []

            embed_signal_length = self.x.shape[-1] - (self.order - 1) * self.delay

            indices = [[(i * self.delay), (i * self.delay + embed_signal_length)] for i in range(self.order)]

            for i in range(self.order):
                # loop with the order
                temp = self.x[:, indices[i][0]: indices[i][1]].reshape(-1, embed_signal_length, 1)
                # slicing the signal with the indices of each order (vectorized operation)

                y.append(temp)
                # append the sliced signal to list

            y = np.concatenate(y, axis=-1)
            # print(np.argpartition(y, kth=3, axis=1))
            # y = np.apply_along_axis(np.argsort, 2, y )
            y = np.argpartition(y, kth=1, axis=2)
            return y

    def _gen_frq(self):
        num_per_permutation = np.zeros((self.x.shape[0], len(self.all_perms)))
        for i in range(self.x.shape[0]):
            for j, x_i in enumerate(self.embed[i, ...]):
                for k, perm_k in enumerate(self.all_perms):
                    if tuple(x_i) == perm_k:
                        num_per_permutation[i, k] += 1
        return num_per_permutation

    def gen_prob(self):
        return self.frqs / np.sum(self.frqs, axis=1)[0]

    def entropy(self, x: np.ndarray, base=2):
        """Returns x log_b x if x is positive, 0 if x == 0, and np.nan
        otherwise. This handles the case when the power spectrum density
        takes any zero value.
        """
        xlogx = np.zeros(x.shape)
        xlogx[x < 0] = np.nan
        valid = x > 0
        xlogx[valid] = x[valid] * np.log(x[valid]) / np.log(base)

        return -1 * xlogx.sum(axis=1) / np.log2(math.factorial(self.order))


# a = np.arange(1, 2100)
# np.random.shuffle(a)
# np.random.shuffle(a)
#
# b = np.arange(1, 2100)
# b = b[::-1]
# # np.random.shuffle(b)
# c = np.arange(1, 2100)
# np.random.shuffle(c)
# x = np.array([a, b[::-1],b, c, c])
#
# order = 3
# perm = Permutation(x, order, 1)
# print(x)
# print(perm.embed)
# print(perm.all_perms)
# print(perm._gen_frq())
# print(perm.gen_prob())
# print(perm.entropy(perm.gen_prob()))
#
# print(np.partition([4,3,2,1],0))
# print(np.partition([4,3,2,1],1))
# print(np.partition([4,3,2,1],2))
# print(np.partition([4,3,2,1],3))

# print(np.array([4,3,2,1])[np.argpartition([4,3,2,1], kth=order - 1)])

#
#
# order = 3
# perms = list(itertools.permutations(np.arange(0, order), order))
# for k, prm_i in enumerate(perms):
#     for j, prm_j in enumerate(perms):
#         if prm_i[0] == prm_j[0] == 0:
#             # for p in range(1,len(prm_j)):
#             if all(prm_j[i] <= prm_j[i + 1] for i in range(len(prm_j) - 1)):
#                 # if prm_i[-1] == order-1:
#                 fig, axs = plt.subplots(2)
#                 plt.figure(j + k + 1)
#                 axs[0].plot(prm_i, '.-', c='r')
#                 axs[1].plot(prm_j, '.-', c='b')
#
#         elif prm_i[0] == prm_j[0] == order - 1:
#             # for p in range(1,len(prm_j)):
#             if all(prm_i[i] >= prm_i[i + 1] for i in range(len(prm_j) - 1)):
#                 fig, axs = plt.subplots(2)
#                 plt.figure(j + k + 1000)
#                 axs[0].plot(prm_i, '.-', c='r')
#                 axs[1].plot(prm_j, '.-', c='b')
#
# plt.show()


def micro_channels(order=3):
    channels = []
    perms = list(itertools.permutations(np.arange(0, order), order))
    for k, prm_i in enumerate(perms):
        for j, prm_j in enumerate(perms):
            if prm_i[0] == prm_j[0] == 0:
                if all(prm_j[i] <= prm_j[i + 1] for i in range(len(prm_j) - 1)):
                    channels.append([prm_i, prm_j])
            elif prm_i[0] == prm_j[0] == order - 1:
                if all(prm_i[i] >= prm_i[i + 1] for i in range(len(prm_j) - 1)):
                    channels.append([prm_i, prm_j])
    return channels

import matplotlib.pyplot as plt

chans = micro_channels(4)
for i in range(1,7):
    print(micro_channels(i))