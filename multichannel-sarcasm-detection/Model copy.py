# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from transformers import AutoModel
from GarphModel import ADGCN  # nama file asli: GarphModel.py (typo pada repo)


class dualModel(nn.Module):
    """
    Multichannel Sarcasm Detection Model.

    Architecture:
        - IndoBERTlite encoder  → CLS token vector (768-dim)
        - ADGCN graph encoder   → graph-aware representation (dim_sen-dim)
        - Fusion: concat [bert_rep, adgcn_rep]  →  dense classifier (2 classes)
    """

    # Dimensi output CLS IndoBERTlite (base variant → 768)
    BERT_HIDDEN_DIM = 768

    def __init__(self, opt, n_vocab, embed_list):
        super(dualModel, self).__init__()

        self.device = opt.device
        self.t_sne  = opt.t_sne  # selalu berpasangan dengan mode prediksi

        # ------------------------------------------------------------------ #
        # 1. IndoBERT-base encoder (fine-tune dengan LR sangat kecil)
        # ------------------------------------------------------------------ #
        self.encoder = AutoModel.from_pretrained(
            "indobenchmark/indobert-base-p1"
        )

        # ------------------------------------------------------------------ #
        # 2. ADGCN – graph encoder berbasis dependency & sentic graph
        # ------------------------------------------------------------------ #
        self.add_module('adgcn', ADGCN(opt, n_vocab, embed_list))

        # Dimensi output ADGCN:
        # Di GarphModel.py, LSTM internal ADGCN di-hardcode bidirectional=True (baris 46).
        # Output akhir diambil dari text_out (BiLSTM) bukan GCN → selalu dim_hidden * 2.
        dim_adgcn = opt.dim_hidden * 2

        # ------------------------------------------------------------------ #
        # 3. Dense classifier
        #    Input : concat [CLS(768), adgcn(dim_adgcn)]
        #    Output: 2 kelas (non-sarcasm / sarcasm)
        # ------------------------------------------------------------------ #
        dense_input_dim = self.BERT_HIDDEN_DIM + dim_adgcn
        self.dense = nn.Sequential(
            nn.Linear(dense_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 2)           # logit mentah (tanpa softmax) → pakai FocalLoss
        )

        print(f'[dualModel] BERT_dim={self.BERT_HIDDEN_DIM}  '
              f'ADGCN_dim={dim_adgcn}  '
              f'dense_input_dim={dense_input_dim}')
        print('****parameter all set****')

    def forward(self, dict_inst):
        """
        Args:
            dict_inst (dict): wajib berisi kunci:
                - 'input_ids'       : LongTensor [B, seq_len]
                - 'attention_mask'  : LongTensor [B, seq_len]
                - 'sens'            : LongTensor [B, max_len]  (untuk ADGCN)
                - 'len_sen'         : np.ndarray panjang tiap kalimat
                - 'dependency_graph': FloatTensor [B, max_len, max_len]
                - 'sentic_graph'    : FloatTensor [B, max_len, max_len]

        Returns:
            prob       : FloatTensor [B, 2]  – probabilitas kelas (softmax)
        """

        # --- IndoBERTlite encoding -------------------------------------------
        bert_output = self.encoder(
            input_ids      = dict_inst['input_ids'],
            attention_mask = dict_inst['attention_mask']
        )
        # Ambil vektor [CLS] dari last_hidden_state[:, 0, :]  → [B, 768]
        bert_rep = bert_output.last_hidden_state[:, 0, :]

        # --- ADGCN encoding --------------------------------------------------
        adgcn_rep = self.adgcn(
            dict_inst['sens'],
            dict_inst['len_sen'],
            dict_inst['dependency_graph'],
            dict_inst['sentic_graph']
        )  # → [B, dim_adgcn]

        # --- Fusion ----------------------------------------------------------
        dense_input = torch.cat([bert_rep, adgcn_rep], dim=-1)  # [B, 768+dim_adgcn]

        # --- Classification --------------------------------------------------
        logits = self.dense(dense_input)                    # [B, 2]  logit mentah
        prob   = torch.softmax(logits, dim=-1)              # [B, 2]  probabilitas

        if self.t_sne:
            return prob, logits, bert_rep, adgcn_rep, dense_input
        else:
            return prob, logits
