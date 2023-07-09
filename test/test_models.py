import os
import sys
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch.nn.functional as F
import torch
from torch import optim, nn
from torch_geometric.data import DataLoader # !
from tqdm import tqdm
from src.nn.model import MLP, AutoEncoder, GAT, Node2Vec
from src.config import Config
import json

if __name__ == '__main__':
    config = Config(default_config_file='configs/darts_pretraining.yaml')
    args = config.load_config()
    # model = AutoEncoder(
    #     input_dim=args.auto_model_in_channels,
    #     emb_dim=args.auto_model_emb_dim,
    # )
    # ########################Train########################
    # N, D = 32, 256
    # x = torch.randn(N, D)
    # print(model)
    # print(model(x).shape)
    # ########################Eval#########################
    # print(model._return_embedding)
    # model.set_return_embedding(True)
    # print(model._return_embedding)

    # with torch.no_grad():
    #     emb = model(x)
    # print(emb.shape)    
    # #####################################################

    # ########################Train########################
    # with open('data/darts-json-100', 'r') as f:
    #     data = json.load(f)
    with open('data/baseline/graph-102722run.json', 'r') as f:
        data = json.load(f)
    # print(data.keys())
    print(data['edge_index'])
    model = GAT(in_channels=256, hidden_channels=256, num_layers=1, num_regs=3, num_classes=3,)
    # model = Node2Vec()
    
    # print(model)
    # print(model(x).shape)

    # model = GraphNetwork(in_channelshidden_channels=hidden_dim_array[0], hidden_channels=hidden_dim_array[0], num_layers=1)

    # torch.Size([32, 256])
    # False
    # True
    # torch.Size([32, 64])    