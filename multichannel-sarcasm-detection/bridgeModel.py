# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from transformers import AutoTokenizer

from basicModel import Lang
from Model import dualModel


# Parameter name patterns yang TIDAK dikenakan weight decay (HF Trainer convention)
NO_DECAY_PATTERNS = ['bias', 'LayerNorm.weight', 'layer_norm.weight']


def _split_decay(named_params, no_decay_patterns=NO_DECAY_PATTERNS):
    """Pisahkan named params jadi 2 list: (decay, no_decay)."""
    decay   = [p for n, p in named_params
               if not any(nd in n for nd in no_decay_patterns)]
    nodecay = [p for n, p in named_params
               if     any(nd in n for nd in no_decay_patterns)]
    return decay, nodecay


# ============================================================================ #
#  Focal Loss
# ============================================================================ #

class FocalLoss(nn.Module):
    """
    Focal Loss untuk menangani class imbalance dan hard examples.

    Persamaan:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        gamma (float): Fokus pada hard examples. Default 2.0.
        alpha (float | list | None):
            - float          : bobot untuk kelas positif (kelas 1 / sarkasme).
              Kelas 0 mendapat bobot (1 - alpha).
            - list/Tensor    : bobot per-kelas, panjang = n_class.
            - None           : tanpa pembobotan (setiap kelas bobot 1).
        reduction (str): 'mean' | 'sum' | 'none'.
    """

    def __init__(self, gamma: float = 2.0, alpha=0.25, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma     = gamma
        self.reduction = reduction

        # Simpan alpha sebagai tensor bobot per-kelas
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            # alpha adalah bobot kelas-1 (sarkasme); kelas-0 mendapat (1-alpha)
            self.register_buffer('alpha', torch.tensor([1.0 - alpha, alpha]))
        else:
            # list / Tensor berisi bobot tiap kelas secara eksplisit
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : FloatTensor [B, C] – logit mentah (belum softmax).
            targets : LongTensor  [B]   – indeks kelas ground-truth.

        Returns:
            Scalar loss (sesuai `reduction`).
        """
        # Hitung log-softmax & softmax sekali saja
        log_prob = F.log_softmax(logits, dim=-1)           # [B, C]
        prob     = torch.exp(log_prob)                      # [B, C]

        # Kumpulkan probabilitas kelas yang benar → p_t
        log_pt = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
        pt     = prob.gather(1, targets.unsqueeze(1)).squeeze(1)      # [B]

        # Fokus modulator: (1 - p_t)^γ
        focal_weight = (1.0 - pt) ** self.gamma                       # [B]

        # Terapkan alpha
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]            # [B]
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_pt                                  # [B]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ============================================================================ #
#  bridgeModel
# ============================================================================ #

class bridgeModel(nn.Module):
    """
    Bridge antara dataloader dan dualModel.
    Menangani:
        - Tokenisasi IndoBERT (via AutoTokenizer)
        - Penyiapan tensor ADGCN
        - Training step dengan FocalLoss
    """

    def __init__(self, FLAGS, vocab=None, embed=None):
        super(bridgeModel, self).__init__()

        self.device       = FLAGS.device
        self.max_length_sen = FLAGS.max_length_sen
        self.n_class      = FLAGS.n_class
        self.learning_rate = FLAGS.learning_rate
        self.batch_size   = FLAGS.batch_size
        self.t_sne        = FLAGS.t_sne

        # fp16 mixed-precision (autocast + GradScaler) — match baseline id_sarcasm
        self.use_fp16 = (
            getattr(FLAGS, 'use_fp16', False)
            and torch.cuda.is_available()
            and self.device.type == 'cuda'
        )
        self.scaler = GradScaler() if self.use_fp16 else None

        # ------------------------------------------------------------------ #
        # Lang – masih dipertahankan untuk keperluan ADGCN (word-index)
        # ------------------------------------------------------------------ #
        self.lang = Lang(vocab)

        # ------------------------------------------------------------------ #
        # IndoBERT-base tokenizer
        # ------------------------------------------------------------------ #
        self.tokenizer = AutoTokenizer.from_pretrained(
            "indobenchmark/indobert-base-p1"
        )

        # ------------------------------------------------------------------ #
        # Model utama
        # ------------------------------------------------------------------ #
        self.model = dualModel(FLAGS, len(vocab), embed)
        self.model.to(self.device)

        # ------------------------------------------------------------------ #
        # Focal Loss  (γ=2, α=0.75 → bobot lebih tinggi ke kelas sarkasme/minority;
        # alpha adalah bobot kelas-1, kelas-0 mendapat 1-alpha=0.25)
        # ------------------------------------------------------------------ #
        self.criterion = FocalLoss(gamma=2.0, alpha=0.75, reduction='mean')

        # ------------------------------------------------------------------ #
        # Optimizer – differential learning rate + WD scope ala HF Trainer:
        #   • encoder (IndoBERT)  → LR 1e-5
        #   • ADGCN + Dense       → LR 1e-3
        #   • bias + LayerNorm    → weight_decay = 0.0  (skip decay)
        #   • param lain          → weight_decay = FLAGS.weight_decay
        # ------------------------------------------------------------------ #
        wd = getattr(FLAGS, 'weight_decay', 0.01)

        encoder_named = list(self.model.encoder.named_parameters())
        encoder_ids   = set(map(id, [p for _, p in encoder_named]))
        other_named   = [(n, p) for n, p in self.model.named_parameters()
                         if id(p) not in encoder_ids]

        enc_decay, enc_nodecay = _split_decay(encoder_named)
        oth_decay, oth_nodecay = _split_decay(other_named)

        self.optimizer = optim.AdamW([
            {'params': enc_decay,   'lr': 1e-5, 'weight_decay': wd},
            {'params': enc_nodecay, 'lr': 1e-5, 'weight_decay': 0.0},
            {'params': oth_decay,   'lr': 1e-3, 'weight_decay': wd},
            {'params': oth_nodecay, 'lr': 1e-3, 'weight_decay': 0.0},
        ])

        print('[bridgeModel] Optimizer ready – '
              f'encoder LR=1e-5, other LR=1e-3, WD={wd} (skip bias+LayerNorm), '
              f'fp16={self.use_fp16}')

    # ---------------------------------------------------------------------- #
    # Data preprocessing
    # ---------------------------------------------------------------------- #

    def gen_batch_data(self, batched_data: dict) -> dict:
        """
        Mengubah raw batch menjadi tensor siap pakai.

        Kunci yang WAJIB ada di `batched_data`:
            - 'sentences'          : list[str]       – kalimat mentah
            - 'dependency_graphs'  : list[np.ndarray]
            - 'sentic_graphs'      : list[np.ndarray]
            - 'sarcasms'           : list[int]        – label (0/1)

        Kunci opsional (untuk ADGCN word-index):
            - 'length_sen'         : list[int]

        Returns:
            dict dengan kunci:
                'input_ids', 'attention_mask',
                'sens', 'len_sen',
                'dependency_graph', 'sentic_graph',
                'sarcasms'
        """
        dict_data = {}

        # --- IndoBERT tokenization --------------------------------------- #
        encoding = self.tokenizer(
            batched_data['sentences'],
            padding      = 'max_length',
            truncation   = True,
            max_length   = self.max_length_sen,
            return_tensors = 'pt'
        )
        dict_data['input_ids']      = encoding['input_ids'].to(self.device)
        dict_data['attention_mask'] = encoding['attention_mask'].to(self.device)

        # --- ADGCN word-index input ----------------------------------------- #
        # Lang.VariablesFromSentences menghasilkan LongTensor [B, max_len]
        dict_data['sens']    = self.lang.VariablesFromSentences(
            batched_data['sentences'], True, self.device
        )
        dict_data['len_sen'] = batched_data['length_sen']

        # --- Graph matrices -------------------------------------------------- #
        dict_data['dependency_graph'] = torch.FloatTensor(
            batched_data['dependency_graphs']
        ).to(self.device)
        dict_data['sentic_graph'] = torch.FloatTensor(
            batched_data['sentic_graphs']
        ).to(self.device)

        # --- Label ----------------------------------------------------------- #
        sarcasms = torch.LongTensor(batched_data['sarcasms'])
        dict_data['sarcasms'] = sarcasms.to(self.device)

        return dict_data

    # ---------------------------------------------------------------------- #
    # Inference / Prediction
    # ---------------------------------------------------------------------- #

    def predict(self, batched_data: dict):
        """
        Returns:
            label_idx : list[int]  – prediksi kelas tiap sampel
            (+ representasi t-SNE jika self.t_sne == True)
        """
        self.model.eval()
        b_data = self.gen_batch_data(batched_data)

        autocast_ctx = autocast('cuda') if self.use_fp16 else nullcontext()

        with torch.no_grad(), autocast_ctx:
            if self.t_sne:
                prob, logits, bert_rep, adgcn_rep, dense_input = self.model(b_data)
            else:
                prob, logits = self.model(b_data)

        label_idx = [tmp.item() for tmp in torch.argmax(prob, dim=-1)]

        if self.t_sne:
            return label_idx, bert_rep, adgcn_rep, dense_input
        else:
            return label_idx

    # ---------------------------------------------------------------------- #
    # Training Step
    # ---------------------------------------------------------------------- #

    def stepTrain(self, batched_data: dict, inference: bool = False):
        """
        Satu langkah training (atau evaluasi jika inference=True).

        Loss: FocalLoss(γ=2, α=0.25) pada prediksi kelas utama saja.

        Returns:
            loss_val  : float         – nilai Focal Loss
            prob_np   : np.ndarray    – probabilitas [B, 2]
        """
        if inference:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()

        b_data = self.gen_batch_data(batched_data)

        # autocast aktif kalau fp16 ON (training & inference) — match HF Trainer fp16
        autocast_ctx = autocast('cuda') if self.use_fp16 else nullcontext()
        # inference jalan dalam no_grad supaya hemat memori
        nograd_ctx   = torch.no_grad() if inference else nullcontext()

        with nograd_ctx, autocast_ctx:
            if self.t_sne:
                prob, logits, bert_rep, adgcn_rep, dense_input = self.model(b_data)
            else:
                prob, logits = self.model(b_data)
            # Focal Loss – pada prediksi kelas utama (sarcasm vs. non-sarcasm)
            loss = self.criterion(logits, b_data['sarcasms'])

        if not inference:
            if self.use_fp16:
                self.scaler.scale(loss).backward()
                # unscale dulu sebelum grad clipping (HF convention)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

        loss_val = loss.item()
        prob_np  = prob.detach().float().cpu().numpy()  # .float() supaya softmax fp16 → fp32

        return loss_val, prob_np

    # ---------------------------------------------------------------------- #
    # Save / Load
    # ---------------------------------------------------------------------- #

    def save_model(self, dir: str, idx):
        os.makedirs(dir, exist_ok=True)
        torch.save(self.state_dict(), f'{dir}/model{idx}.pth')
        print('****save state dict****')
        print(list(self.state_dict().keys())[:10], '...')

    def load_model(self, dir: str, idx: int = -1, device: str = 'cpu'):
        if idx < 0:
            params = torch.load(dir, map_location=device)
            self.load_state_dict(params)
        else:
            print('****load state dict****')
            self.load_state_dict(
                torch.load(f'{dir}/model{idx}.pth', map_location=device)
            )