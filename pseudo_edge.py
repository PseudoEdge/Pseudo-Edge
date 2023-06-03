import torch
import numpy as np
import scipy.sparse as ssp
from torch_sparse import SparseTensor
import pdb
import itertools
import random
import os, inspect
import pickle
from torch_geometric.utils.dropout import dropout_adj
import itertools
import multiprocessing
from multiprocessing import set_start_method
import time

device = torch.device('cuda')

def get_local_edges(pool_idx, edges, tmp1s, tmp2s, args, len_one_hops):
    all_edges = []
    ptrs = []
    subptrs = []
    total_sum = 0
    for idx in range(len(edges)):
        if pool_idx == 0 and idx % 500 == 0:
            print(idx)
        edge = edges[idx]
        tmp1 = tmp1s[idx]
        tmp2 = tmp2s[idx]
        len_one_hop = len_one_hops[idx]

        tuples = list(itertools.product([edge[0].item()], tmp2)) # i x N_j
        tuples.extend(list(itertools.product([edge[1].item()], tmp1))) # j x N_i
        tuples.extend(list(itertools.product(tmp1, tmp2))) # N_i x N_j
        tuples = np.array(tuples)
        self_loops = np.nonzero(tuples[:, 0] == tuples[:, 1])[0]
        
        loop_mask = np.ones(tuples.shape, dtype=np.bool)
        loop_mask[self_loops] = False
        len_one_hop_cp = len_one_hop

        _, uniques = np.unique(tuples, axis=0, return_index=True)
        overlaps = list(set(range(tuples.shape[0])) - set(uniques))
        len_one_hop -= len(np.nonzero(np.array(overlaps) < len_one_hop)[0])
        overlap_mask = np.zeros(tuples.shape, dtype=np.bool)
        overlap_mask[uniques] = True
        tuple_mask = np.logical_and(loop_mask, overlap_mask)
        tuples = tuples[tuple_mask].reshape(-1, 2)

        len_one_hop -= len(np.nonzero(self_loops < len_one_hop_cp)[0])
        tuples = [(edge1, edge2) for edge1, edge2 in tuples]
        
        if len(tuples) > args.max_edges:
            if len_one_hop > args.max_edges*0.5:
                one_hop = random.sample(tuples[:len_one_hop], int(args.max_edges*0.5))
                two_hop = random.sample(tuples[len_one_hop:], int(args.max_edges*0.5))
            else:
                one_hop = tuples[:len_one_hop]
                two_hop = random.sample(tuples[len_one_hop:], min(len(tuples)-len_one_hop, len_one_hop))
            tuples = one_hop + two_hop
        else:
            m = min(len(tuples)-len_one_hop, len_one_hop)
            if m == 0:
                one_hop = tuples[:len_one_hop]
                two_hop = random.sample(tuples[len_one_hop:], m)
                tuples = one_hop + two_hop
            else:
                one_hop = tuples[:m]
                two_hop = random.sample(tuples[m:], m)
                tuples = one_hop + two_hop

        if len(tuples) != 0:
            all_edges.append(torch.from_numpy(np.array(tuples)))
            assert len(tuples) != 0
            total_sum += len(tuples)
            ptrs.append(total_sum)
            subptrs.append(len(one_hop))
        else:
            ptrs.append(total_sum)
            subptrs.append(0)

    with open('pool_tmp/'+args.dataset+'_'+args.gnn+'_'+str(pool_idx)+'.pkl', 'wb') as f:
        pickle.dump([all_edges, ptrs, subptrs], f)

    return 0

def generate_pseudo_edges(model, predictor, adj_t, edge_index, data, split_edge, candidate_edges=None, args=None, num_nodes=None):
    model.eval()
    if predictor is not None:
        predictor.eval()
    
    edge_index = edge_index.cpu()
    with torch.no_grad():
        h = model(data.x, data.adj_t)
            
        interval = 2000000
        for i in range(0, candidate_edges.shape[-1], interval):
            if i == 0:
                out = predictor(h[candidate_edges[0, i:i+interval]], h[candidate_edges[1, i:i+interval]])
            else:
                out = torch.cat((out, predictor(h[candidate_edges[0, i:i+interval]], h[candidate_edges[1, i:i+interval]])))
        out = out.squeeze()

        vars = None
        if args.uncertainty:
            outs = []
            for iter in range(10):
                tmp_adj = dropout_adj(edge_index, p=0.5)[0]
                h = model(data.x, tmp_adj.to(device))

                interval = 2000000
                for i in range(0, candidate_edges.shape[-1], interval):
                    if i == 0:
                        tmp_out = predictor(h[candidate_edges[0, i:i+interval]], h[candidate_edges[1, i:i+interval]])
                    else:
                        tmp_out = torch.cat((tmp_out, predictor(h[candidate_edges[0, i:i+interval]], h[candidate_edges[1, i:i+interval]])))
                outs.append(tmp_out.squeeze().cpu())
            vars = torch.stack(outs).var(dim=0)
            
            del outs
            torch.cuda.empty_cache()
            
            pseudo_edges, candidate_edges = filter_edges_confidence_score(data, adj_t, out, predictor, h, candidate_edges, args, vars=vars, split_edge=split_edge)     

    return pseudo_edges, candidate_edges
    

def filter_edges_confidence_score(data, adj_t, out, predictor, h, edges, args, vars=None, split_edge=None):
    indices = torch.topk(out, int(args.num_train_edges*args.add_ratio)*2)[1]
    mask = torch.zeros(len(out), dtype=torch.bool)
    mask[indices] = True
    filter_mask = torch.logical_and(mask, (out>args.threshold).cpu())
    if args.uncertainty:
        filter_mask = torch.logical_and(filter_mask, (vars < args.uncertainty_thr).cpu())

    if args.global_ratio != 0:
        print('global_ratio: ', args.global_ratio)
        global_indices = torch.topk(out, int(args.num_train_edges*args.add_ratio*args.global_ratio))[1]
        global_mask = torch.zeros(len(out), dtype=torch.bool)
        global_mask[global_indices] = True
        global_mask = torch.logical_and(global_mask, (out>args.threshold).cpu())
        filter_mask = torch.logical_and(filter_mask, ~global_mask)
        
        if args.uncertainty:
            global_mask = torch.logical_and(global_mask, (vars < args.uncertainty_thr).cpu()) 
        global_edges_filtered = edges[:,global_mask]

    if args.global_ratio != 1:
        edges_filtered = edges[:,filter_mask]
        filter_mask = filter_mask.to(device)
        out_filtered = out[filter_mask]

        all_edges = []
        ptrs = [0]
        subptrs = []
        total_sum = 0
        n_edges = edges_filtered.shape[1]-1
        n_process = 64
        pool = multiprocessing.Pool(processes=n_process)
        pool_edges_filtered = edges_filtered.cpu().t()
        chunk_size = int(len(pool_edges_filtered) / n_process)
        
        total_tmp1s = []
        total_tmp2s = []
        total_len_one_hop = []
        for i in range(0, edges_filtered.shape[-1]):
            edge = pool_edges_filtered[i]
            tmp1 = adj_t[edge[0].item()].storage.col().tolist()
            tmp2 = adj_t[edge[1].item()].storage.col().tolist()
            total_tmp1s.append(tmp1)
            total_tmp2s.append(tmp2)
            total_len_one_hop.append(len(tmp1)+len(tmp2))
        
        pool_tmp1s = []
        pool_tmp2s = []
        pool_len_one_hop = []
        pool_args = []
        pool_edges = []
        pool_idxs = []
        pool_idx = 0
        for i in range(0, edges_filtered.shape[-1], chunk_size):
            pool_tmp1s.append(total_tmp1s[i:i+chunk_size])
            pool_tmp2s.append(total_tmp2s[i:i+chunk_size])
            pool_len_one_hop.append(total_len_one_hop[i:i+chunk_size])
            pool_edges.append(pool_edges_filtered[i:i+chunk_size])
            pool_args.append(args)
            pool_idxs.append(pool_idx)
            pool_idx += 1
        pool_result = pool.starmap(get_local_edges, zip(pool_idxs, pool_edges, pool_tmp1s, pool_tmp2s, pool_args, pool_len_one_hop))
        pool.close()
        
        all_edges = []
        ptrs = [0]
        subptrs = []
        for pool_idx in range(len(pool_edges)):
            with open('pool_tmp/'+args.dataset+'_'+args.gnn+'_'+str(pool_idx)+'.pkl', 'rb') as f:
                tmp = pickle.load(f)
            all_edges.extend(tmp[0])
            ptrs.extend(tmp[1])
            subptrs.extend(tmp[2])
    
        all_edges = torch.cat(all_edges, dim=0)
        batch_size = 1000000
        total_sum = 0
        result_tmp = []
        while total_sum < len(all_edges):
            result_tmp.append(predictor(h[all_edges[total_sum:total_sum+batch_size,0]], h[all_edges[total_sum:total_sum+batch_size,1]]).squeeze().cpu())
            total_sum += batch_size
        result_tmp = torch.cat(result_tmp)

        local_scores = []
        for i, ptr in enumerate(ptrs):
            if i != len(ptrs)-1:
                onehop = result_tmp[ptrs[i]:ptrs[i+1]][:subptrs[i]]
                twohop = result_tmp[ptrs[i]:ptrs[i+1]][subptrs[i]:]

                if args.remove_sparse != 0:
                    if len(result_tmp[ptrs[i]:ptrs[i+1]]) <= args.remove_sparse:
                        local_scores.append(float('inf'))
                    else:
                        local_scores.append(args.cc_ratio * onehop.mean() + (1-args.cc_ratio) * twohop.mean())
                else:
                    if len(result_tmp[ptrs[i]:ptrs[i+1]]) == 0:
                        local_scores.append(float('inf'))
                    elif len(twohop) == 0:
                        if args.remove_empty_twohops:
                            local_scores.append(float('inf'))
                        else:
                            local_scores.append(onehop.mean())
                    else:
                        local_scores.append(args.cc_ratio * onehop.mean() + (1-args.cc_ratio) * twohop.mean())

        mask_l = torch.zeros(len(out_filtered), dtype=torch.bool)
        local_scores = torch.tensor(local_scores, dtype=out_filtered.dtype).to(device)
        local_ratio = int(args.num_train_edges*args.add_ratio*(1-args.global_ratio))

        gaps = out_filtered - local_scores
        if local_ratio < len(gaps):
            indices_l = torch.topk(gaps, local_ratio)[1]
            print('==========================')
            print(gaps.shape)
            print(indices_l.shape)
            mask_l[indices_l] = True
        else:
            mask_l[:] = True
    
    if args.global_ratio == 0:
        pseudo_edges = edges_filtered[:,mask_l]
    elif args.global_ratio == 1:
        pseudo_edges = global_edges_filtered
    else:
        local_edges_filtered = edges_filtered[:,mask_l]
        pseudo_edges = torch.cat([global_edges_filtered, local_edges_filtered], dim=-1)
        print(f'global threshold result: {edges_filtered.shape}')
    print(f'overall threshold results: {pseudo_edges.shape}')

    pseudo_edges = pseudo_edges.t().cpu()

    if args.global_ratio < 1:
        indices = torch.nonzero(filter_mask == True).reshape(-1)
        indices_lf = indices[mask_l==False]
        mask_lf = torch.ones(out.shape, dtype=torch.bool).to(device)
        mask_lf[indices_lf] = False
        filter_mask = torch.logical_and(filter_mask, mask_lf).cpu()
        if args.global_ratio > 0:
            filter_mask = torch.logical_or(filter_mask, global_mask)
    else:
        filter_mask = global_mask

    edges = edges[:,filter_mask==False]
    
    print('Finished pseudo-labels generation,')

    return pseudo_edges, edges

