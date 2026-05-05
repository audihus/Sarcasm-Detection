# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from DynamicRNN import DynamicRNN
from models.affectivegcn import GraphConvolution  
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class ADGCN(nn.Module):
    def __init__(self, opt, n_vocab, embed_list):
        super(ADGCN, self).__init__()
        self.n_layers = opt.n_layers
        self.dim_input = opt.dim_input
        self.dim_hidden = opt.dim_hidden
        self.bidirectional = True if opt.bidirectional else False  
        self.rnn_type = opt.rnn_type
        self.device = opt.device
        self.f=nn.LSTM(
                input_size=opt.dim_input, hidden_size=opt.dim_hidden, num_layers=1,
                bias=True, batch_first=True, dropout=0, bidirectional=True)  

        # 嵌入层和dropout层
        self.add_module('embed', nn.Embedding(n_vocab, self.dim_input))
        self.add_module('embed_dropout', nn.Dropout(opt.embed_dropout_rate))
        
        # 6层，与AFFGCN保持一致）
       
        
        
        self.add_module('gc1', GraphConvolution(self.dim_hidden*2, self.dim_hidden*2))
        self.add_module('gc2', GraphConvolution(self.dim_hidden*2, self.dim_hidden*2))
        self.add_module('gc3', GraphConvolution(self.dim_hidden*2, self.dim_hidden*2))
        self.add_module('gc4', GraphConvolution(self.dim_hidden*2, self.dim_hidden*2))
        self.add_module('gc5', GraphConvolution(self.dim_hidden*2, self.dim_hidden*2))
        self.add_module('gc6', GraphConvolution(self.dim_hidden*2, self.dim_hidden*2))

        # 初始化嵌入层权重
        self.init_weights(embed_list)

    def init_weights(self, embed_list):
        # 保持原有初始化逻辑
        self.embed.weight.data.copy_(torch.from_numpy(embed_list))

    def forward(self, embed_list, length_list, adj,sentic_adj):
        # 1. 嵌入层+dropout
        embedded = self.embed_dropout(self.embed(embed_list))  # shape: [batch_size, seq_len, dim_input]
        text_out, _ = self.f(embedded)

        # 2. 采用 affectivegcn 的图卷积逻辑：交替使用依赖图和情感图
        # 第一层用依赖图，第二层用情感图，以此类推
        x = F.relu(self.gc1(text_out, adj))
        x = F.relu(self.gc2(x, sentic_adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, sentic_adj))
        x = F.relu(self.gc5(x, adj))
        x = F.relu(self.gc6(x, sentic_adj))

        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1)  
        return x
