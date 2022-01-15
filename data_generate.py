import random
import numpy as np


def get_all_samples(conjunction):
    pos = []
    neg = []
    for index in range(conjunction.shape[0]):
        for col in range(conjunction.shape[1]):
            if conjunction[index, col] == 1:
                pos.append([index, col, 1])
            else:
                neg.append([index, col, 0])
    pos_len = len(pos)
    new_neg = random.sample(neg, pos_len)
    samples = pos + new_neg
    samples = random.sample(samples, len(samples))
    samples = np.array(samples)
    return samples


def generate_f(samples: np.ndarray, features: list):

    vec_lens = np.zeros(len(features), dtype=int)
    for index in range(len(features)):
        vec_lens[index] = features[index].shape[1]
    vec_len = np.sum(vec_lens)
    num = samples.shape[0]
    feature = np.zeros([num, vec_len])
    label = np.zeros([num])

    for i in range(num):
        tail = 0
        for index in range(len(features)):
            head = tail
            tail += vec_lens[index]
            feature[i, head: tail] = features[index][samples[i, 1 if index % 2 else 0], :]
        label[i] = samples[i, 2]
    return feature, label