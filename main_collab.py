import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from logger import Logger
from models import GCN, SAGE, JKNet, LinkPredictor
import pdb
from utils import init_seed, preprocess_hops
import numpy as np
import scipy.sparse as ssp
import os, inspect
import pickle
import torch_sparse
import warnings
from pseudo_edge import generate_pseudo_edges

def train(model, predictor, data, split_edge, optimizer, batch_size):
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(data.x, data.adj_t)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()
    h = model(data.x, data.adj_t)

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)
    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)
    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)
    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)
    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)
    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def main():
    warnings.simplefilter("ignore", UserWarning)
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--gnn', type=str, default='GCN')
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--dataset', type=str, default='collab')
    parser.add_argument('--max_edges', type=int, default=100)
    parser.add_argument('--global_ratio', type=float, default=1.)
    parser.add_argument('--generation_runs', type=int, default=5)
    parser.add_argument('--add_ratio', type=float, default=0.4)
    parser.add_argument('--remove_sparse', type=int, default=0)
    parser.add_argument('--cc_ratio', type=float, default=0.8, 
                    help='ratio for onehop(twohop: 1-args.cc_ratio)')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--uncertainty', action='store_true')
    parser.add_argument('--uncertainty_thr', type=float, default=1e-4)
    parser.add_argument('--num_nodes', type=int)
    parser.add_argument('--preprocess_candidates', action='store_true')
    parser.add_argument('--load_vanilla_model', action='store_true')
    parser.add_argument('--load_pseudo_labels', action='store_true')
    parser.add_argument('--load_pretrained_model', action='store_true')
    parser.add_argument('--pl_dir', type=str, default='best_pseudo_labels')
    parser.add_argument('--best_vanilla_model_dir', type=str, default='best_vanilla_model')
    parser.add_argument('--best_pl_model_dir', type=str, default='best_pl_model')
    parser.add_argument('--best_model_dir', type=str, default='best_model')
    parser.add_argument('--pretrained_model_dir', type=str, default='pretrained_model')

    args = parser.parse_args()
    print(args)

    args.dataset = 'collab'
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-collab')
    data = dataset[0]
    edge_index = data.edge_index
    data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    data = T.ToSparseTensor()(data)
    
    split_edge = dataset.get_edge_split()
    args.num_train_edges = split_edge['train']['edge'].shape[0]

    data.adj_t = SparseTensor.from_edge_index(edge_index).t()
    data.adj_t = data.adj_t.to_symmetric()
    args.num_nodes = data.num_nodes

    data = data.to(device)

    if args.gnn == 'GCN':
        model = GCN(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    elif args.gnn == 'SAGE':
        model = SAGE(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    elif args.gnn == 'JKNet':
        model = JKNet(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout, args=args).to(device)

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-collab')

    # 1. Train GNN only with given labeled data
    if not args.load_vanilla_model:
        init_seed(0)
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr) 
        best_valid_performance = 0
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data, split_edge, optimizer, args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, data, split_edge, evaluator, args.batch_size)
                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train: {100 * train_hits:.2f}%, '
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
                    print('---')
                
                valid_performance = results['Hits@50'][1]
                if valid_performance > best_valid_performance:
                    torch.save(model.state_dict(), os.path.join(args.best_vanilla_model_dir, args.dataset+"_"+args.gnn+"_best_model.pt"))
                    torch.save(predictor.state_dict(), os.path.join(args.best_vanilla_model_dir, args.dataset+"_"+args.gnn+"_best_predictor.pt"))
                    best_valid_performance = valid_performance
        
        
    # 2. Generate Pseudo-labels
    if args.load_pseudo_labels:
        with open(os.path.join(args.pl_dir, args.dataset+"_"+args.gnn+"_"+"pseudo_labels.pkl"), 'rb') as f:
            best_pseudo_labels = pickle.load(f)
    else:
        loggers = {
            'Hits@20': Logger(args.generation_runs, args),
            'Hits@50': Logger(args.generation_runs, args),
            'Hits@100': Logger(args.generation_runs, args),
        }        
        
        if args.preprocess_candidates:
            preprocess_hops(data.adj_t, split_edge, args)

        with open(args.dataset+'_edge_candidates.pkl', 'rb') as f:
            candidate_edges = pickle.load(f)
        if candidate_edges.shape[-1] > 500000000:
            perm = np.random.permutation(np.arange(candidate_edges.shape[-1]))
            candidate_edges = candidate_edges[:,perm[:500000000]]

        best_highest_valid = 0
        best_pseudo_labels = None
        for run in range(args.generation_runs):
            best_valid_performance = 0
            print('Pseudo-label generation')
            print('#################################          ', run, '          #################################')
            if run == 0:
                model.load_state_dict(torch.load(os.path.join(args.best_vanilla_model_dir, args.dataset+"_"+args.gnn+"_"+"best_model.pt")))
                predictor.load_state_dict(torch.load(os.path.join(args.best_vanilla_model_dir, args.dataset+"_"+args.gnn+"_"+"best_predictor.pt")))
            else:
                model.load_state_dict(torch.load(os.path.join(args.best_pl_model_dir, args.dataset+"_"+args.gnn+"_"+"best_model.pt")))
                predictor.load_state_dict(torch.load(os.path.join(args.best_pl_model_dir, args.dataset+"_"+args.gnn+"_"+"best_predictor.pt")))

            print("Start generating pseudo edges!")
            pseudo_edges, candidate_edges = generate_pseudo_edges(model, predictor, data.adj_t, edge_index, data, split_edge, 
                                                                candidate_edges, args=args, num_nodes=args.num_nodes)

            data.adj_t = SparseTensor.from_edge_index(edge_index).t()
            data.adj_t = data.adj_t.to_symmetric()
            data = data.to(device)

            print('#(pseudo_labels): ', pseudo_edges.shape[0])

            tmp = split_edge['train']
            split_edge['train']['edge'] = torch.cat((tmp['edge'],pseudo_edges), dim=0)
            print('#(total labels): ', split_edge['train']['edge'].shape[0])

            init_seed(0)
            model.reset_parameters()
            predictor.reset_parameters()
            optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)        
            for epoch in range(1, 1 + args.epochs):
                loss = train(model, predictor, data, split_edge, optimizer, args.batch_size)

                if epoch % args.eval_steps == 0:
                    results = test(model, predictor, data, split_edge, evaluator, args.batch_size)
                    for key, result in results.items():
                        loggers[key].add_result(run, result)
                    if epoch % args.log_steps == 0:
                        for key, result in results.items():
                            train_hits, valid_hits, test_hits = result
                            print(key)
                            print(f'Run: {run + 1:02d}, '
                                f'Epoch: {epoch:02d}, '
                                f'Loss: {loss:.4f}, '
                                f'Train: {100 * train_hits:.2f}%, '
                                f'Valid: {100 * valid_hits:.2f}%, '
                                f'Test: {100 * test_hits:.2f}%')
                        print('---')
                    
                    valid_performance = results['Hits@50'][1]
                    if valid_performance > best_valid_performance:
                        torch.save(model.state_dict(), os.path.join(args.best_pl_model_dir, args.dataset+"_"+args.gnn+"_"+"best_model.pt"))
                        torch.save(predictor.state_dict(), os.path.join(args.best_pl_model_dir, args.dataset+"_"+args.gnn+"_"+"best_predictor.pt"))
                        best_valid_performance = valid_performance

            for key in loggers.keys():
                print(key)
                final_test, highest_valid = loggers[key].print_statistics(run)          
                if key == 'Hits@50':
                    if best_highest_valid < highest_valid:
                        best_highest_valid = highest_valid
                        best_pseudo_labels = split_edge['train']['edge']

            with open(os.path.join(args.pl_dir, args.dataset+"_"+args.gnn+"_"+"pseudo_labels.pkl"), 'wb') as f:
                pickle.dump(best_pseudo_labels, f)
        

    # 3. Train GNN with best pseudo-labels
    print('')
    print('Train models with the final pseudo-labels')
    split_edge['train']['edge'] = best_pseudo_labels

    if args.load_pretrained_model:
        model.load_state_dict(torch.load(os.path.join(args.pretrained_model_dir, args.dataset+"_"+args.gnn+"_"+"best_model.pt")))
        predictor.load_state_dict(torch.load(os.path.join(args.pretrained_model_dir, args.dataset+"_"+args.gnn+"_"+"best_predictor.pt")))
    
    loggers = {
        'Hits@20': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }      
    for run in range(args.runs):
        best_valid_performance = 0
        print('#################################          ', run, '          #################################')
        init_seed(run)
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data, split_edge, optimizer, args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, data, split_edge, evaluator, args.batch_size)
                for key, result in results.items():
                    loggers[key].add_result(run, result)
                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train: {100 * train_hits:.2f}%, '
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
                    print('---')
                
                valid_performance = results['Hits@50'][1]
                if valid_performance > best_valid_performance:
                    torch.save(model.state_dict(), os.path.join(args.best_model_dir, args.dataset+"_"+args.gnn+"_"+"best_model.pt"))
                    torch.save(predictor.state_dict(), os.path.join(args.best_model_dir, args.dataset+"_"+args.gnn+"_"+"best_predictor.pt"))
                    best_valid_performance = valid_performance

        for key in loggers.keys():
            print(key)
            final_test, highest_valid = loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        final_mean, final_std = loggers[key].print_statistics()


if __name__ == "__main__":
    main()
