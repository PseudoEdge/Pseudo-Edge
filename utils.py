from __future__ import division

import torch
import torch_sparse
from torch_sparse import SparseTensor
import pickle
import numpy as np
import random
import subprocess
from torch_scatter import scatter_add
import pdb
from torch_geometric.utils import degree
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from tqdm import tqdm
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
import math
import os
import scipy.sparse as ssp
from collections import defaultdict
from torch_sparse import sum as sparsesum


def init_seed(seed=2023):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preprocess_hops(A, split_edge, args):
    assert split_edge['valid']['edge'].is_cuda == False
    assert split_edge['test']['edge'].is_cuda == False

    A2 = A @ A
    A2 = torch_sparse.remove_diag(A2)
    row, col, _ = A2.coo()
    edges2 = torch.stack((row, col)).detach().cpu()
    uniq = torch.unique(edges2, dim=1)
    edges2 = uniq
    print('Successfully generated 2-hop edge indices.')
    print('Successfully removed self-loops.')
    e2_v = torch.cat([edges2, split_edge['valid']['edge'].t()], 1)
    uniq, counts = torch.unique(e2_v, dim=1, return_counts=True)
    dup = counts > 1
    edges2 = uniq[:, ~dup]

    e2_v = torch.cat([edges2, split_edge['test']['edge'].t()], 1)
    uniq, counts = torch.unique(e2_v, dim=1, return_counts=True)
    dup = counts > 1
    edges2 = uniq[:, ~dup]

    print('Successfully removed overlaps with valid/test edges.')

    A3 = A2 @ A

    row, col, _ = A3.coo()
    edges3 = torch.stack((row, col)).detach().cpu()
    uniq = torch.unique(edges3, dim=1)
    edges3 = uniq
    print('Successfully generated 3-hop edge indices.')
    self_loops = edges3[0] == edges3[1]
    edges3 = edges3[:, ~self_loops]
    print('Successfully removed self-loops.')

    e3_v = torch.cat([edges3, split_edge['valid']['edge'].t()], 1)
    uniq, counts = torch.unique(e3_v, dim=1, return_counts=True)
    dup = counts > 1
    edges3 = uniq[:, ~dup]

    e3_v = torch.cat([edges3, split_edge['test']['edge'].t()], 1)
    uniq, counts = torch.unique(e3_v, dim=1, return_counts=True)
    dup = counts > 1
    edges3 = uniq[:, ~dup]

    print('Successfully removed overlaps with valid/test edges.')

    cand_edges = torch.cat((edges2, edges3), dim=1)
    cand_edges = torch.unique(cand_edges, dim=1)
    
    with open(args.dataset+'_'+'edge_candidates.pkl', 'wb') as f:
        pickle.dump(cand_edges, f)    

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)