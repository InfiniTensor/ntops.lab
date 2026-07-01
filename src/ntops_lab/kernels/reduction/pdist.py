import torch
from ntops_lab.kernels.reduction.pairwise_distance import run as pairwise_distance

def run(*inputs):
    x, = inputs
    chunks = []
    for i in range(x.shape[0]):
        for j in range(i + 1, x.shape[0]):
            chunks.append(pairwise_distance(x[i:i + 1], x[j:j + 1]))
    return torch.cat(chunks, dim=0)
