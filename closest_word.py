import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F





def nth_closest_vector(data, v, n):
    return data[nth_closest_index(data, v, n)]

def nth_closest_index(data,v, n):
    i = np.argpartition([F.pairwise_distance(v, torch.tensor(w).detach().cpu()) for w in data], n)[n]
    return i

def nth_closest_word(data,v, n):
    return data(nth_closest_index(data, v, n))

def tensor_to_nth_closest_word(words, data, t, n):
    return words[nth_closest_index(data, t, n)]



