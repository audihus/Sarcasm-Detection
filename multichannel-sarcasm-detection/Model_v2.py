# -*- coding: utf-8 -*-
"""
Model_v2.py — dualModelV2
Arsitektur baru: BERT + DepGCN (dep-only) + Pragmatic MLP.

Perubahan dari Model.py:
  - Sentic channel dihapus: dependency_graph dilewatkan ke KEDUA slot adj pada ADGCN.
  - Tambah pragmatic_mlp: 6 -> Linear(6,32) -> ReLU -> Linear(32,32).
  - Dense head: concat[BERT(768) + ADGCN(512) + Prag(32)] = 1312 -> Dropout(0.2) -> Linear(1312,2).
"""
from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from transformers import AutoModel
from GarphModel import ADGCN


class dualModelV2(nn.Module):
    """
    Multichannel sarcasm detection model v2 (dep-only + pragmatic).

    Forward dict_inst keys:
        'input_ids'          : LongTensor  [B, seq_len]
        'attention_mask'     : LongTensor  [B, seq_len]
        'sens'               : LongTensor  [B, max_len]
        'len_sen'            : list[int]
        'dependency_graph'   : FloatTensor [B, max_len, max_len]
        'pragmatic_features' : FloatTensor [B, 6]
    """

    BERT_HIDDEN_DIM = 768

    def __init__(self, opt, n_vocab, embed_list):
        super(dualModelV2, self).__init__()

        self.device = opt.device
        self.t_sne  = opt.t_sne

        # IndoBERT-base encoder
        self.encoder = AutoModel.from_pretrained(
            "indolem/indobertweet-base-uncased"
        )

        # ADGCN: dep-only (dependency_graph passed to both adj slots, sentic dropped)
        self.add_module("adgcn", ADGCN(opt, n_vocab, embed_list))
        dim_adgcn = opt.dim_hidden * 2  # 256*2 = 512

        # Pragmatic MLP: 6-dim handcrafted features -> 32-dim
        self.pragmatic_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        # Dense classifier: 768 + 512 + 32 = 1312 -> 2
        dense_input_dim = self.BERT_HIDDEN_DIM + dim_adgcn + 32
        self.dense = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(dense_input_dim, 2),
        )

        print(
            f"[dualModelV2] BERT={self.BERT_HIDDEN_DIM}  "
            f"ADGCN(dep-only)={dim_adgcn}  "
            f"Pragmatic=32  "
            f"dense_input={dense_input_dim}"
        )

    def forward(self, dict_inst):
        # BERT: CLS pooler_output [B, 768]
        bert_rep = self.encoder(
            input_ids      = dict_inst["input_ids"],
            attention_mask = dict_inst["attention_mask"],
        ).pooler_output

        # ADGCN: dep-only — same matrix for both adj and sentic_adj [B, 512]
        dep_graph = dict_inst["dependency_graph"]
        adgcn_rep = self.adgcn(
            dict_inst["sens"],
            dict_inst["len_sen"],
            dep_graph,  # adj
            dep_graph,  # sentic_adj — sentic channel dropped
        )

        # Pragmatic MLP [B, 32]
        prag_rep = self.pragmatic_mlp(dict_inst["pragmatic_features"])

        # Fusion + classification [B, 1312] -> [B, 2]
        dense_input = torch.cat([bert_rep, adgcn_rep, prag_rep], dim=-1)
        logits = self.dense(dense_input)
        prob   = torch.softmax(logits, dim=-1)

        if self.t_sne:
            return prob, logits, bert_rep, adgcn_rep, dense_input
        return prob, logits
