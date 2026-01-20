import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.append("..")
import d22l as d2l


def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()

    return mask * X / keep_prob


X = torch.arange(16).view(2, 8)
print(dropout(X, 0))
print(dropout(X, 0.5))
print(dropout(X, 1))
