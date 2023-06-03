#!/bin/bash

python main_collab.py --dataset collab --gnn GCN --load_vanilla_model --load_pseudo_labels

# Using pretrained model
# python main_collab.py --dataset collab --gnn GCN --load_vanilla_model --load_pseudo_labels --load_pretrained_model