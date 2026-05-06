# -*- coding: utf-8 -*-
"""
bridgeModel_v2.py
Bridge untuk dualModelV2: dep-only ADGCN + pragmatic channel.

Perubahan dari bridgeModel.py:
  - Import dualModelV2 dari Model_v2.
  - gen_batch_data: hapus sentic_graph, tambah pragmatic_features.
  - Loss: CrossEntropyLoss(weight=[1.0, cw]) default; FocalLoss jika use_focal.
  - Optimizer identik: 4-group AdamW, LR 1e-5/1e-3, WD skip bias+LayerNorm.
"""
from __future__ import unicode_literals, print_function, division

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
from Model_v2 import dualModelV2


NO_DECAY_PATTERNS = ["bias", "LayerNorm.weight", "layer_norm.weight"]


def _split_decay(named_params, no_decay_patterns=NO_DECAY_PATTERNS):
    decay   = [p for n, p in named_params if not any(nd in n for nd in no_decay_patterns)]
    nodecay = [p for n, p in named_params if     any(nd in n for nd in no_decay_patterns)]
    return decay, nodecay


class FocalLoss(nn.Module):
    """Focal Loss (kept for --loss_type focal support)."""

    def __init__(self, gamma: float = 2.0, alpha=0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if isinstance(alpha, float):
            self.register_buffer("alpha", torch.tensor([1.0 - alpha, alpha]))
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=-1)
        prob     = torch.exp(log_prob)
        log_pt   = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt       = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        fw       = (1.0 - pt) ** self.gamma
        if self.alpha is not None:
            fw = self.alpha.to(logits.device)[targets] * fw
        loss = -fw * log_pt
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class bridgeModelV2(nn.Module):
    """
    Bridge antara dataloader dan dualModelV2.

    Batch dict yang diharapkan di gen_batch_data:
        'sentences_raw'      : list[str]
        'sentences'          : list[list[str]]
        'length_sen'         : list[int]
        'dependency_graphs'  : np.ndarray [B, P, P]
        'pragmatic_features' : np.ndarray [B, 6]
        'sarcasms'           : list[int]
    """

    def __init__(self, FLAGS, vocab=None, embed=None):
        super().__init__()

        self.device         = FLAGS.device
        self.max_length_sen = FLAGS.max_length_sen
        self.n_class        = FLAGS.n_class
        self.learning_rate  = FLAGS.learning_rate
        self.batch_size     = FLAGS.batch_size
        self.t_sne          = FLAGS.t_sne

        self.use_fp16 = (
            getattr(FLAGS, "use_fp16", False)
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        )
        self.scaler = GradScaler() if self.use_fp16 else None

        self.lang      = Lang(vocab)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "indobenchmark/indobert-base-p1"
        )

        self.model = dualModelV2(FLAGS, len(vocab), embed)
        self.model.to(self.device)

        use_focal = getattr(FLAGS, "use_focal", False)
        if use_focal:
            self.criterion = FocalLoss(gamma=2.0, alpha=0.75, reduction="mean")
            loss_name = "FocalLoss(gamma=2, alpha=0.75)"
        else:
            cw     = float(getattr(FLAGS, "class_weight_sarcasm", 3.0))
            weight = torch.tensor([1.0, cw]).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weight)
            loss_name = f"CrossEntropyLoss(weight=[1.0, {cw}])"

        wd = getattr(FLAGS, "weight_decay", 0.01)
        encoder_named = list(self.model.encoder.named_parameters())
        encoder_ids   = set(map(id, [p for _, p in encoder_named]))
        other_named   = [
            (n, p) for n, p in self.model.named_parameters()
            if id(p) not in encoder_ids
        ]
        enc_decay, enc_nodecay = _split_decay(encoder_named)
        oth_decay, oth_nodecay = _split_decay(other_named)

        self.optimizer = optim.AdamW([
            {"params": enc_decay,   "lr": 1e-5, "weight_decay": wd},
            {"params": enc_nodecay, "lr": 1e-5, "weight_decay": 0.0},
            {"params": oth_decay,   "lr": 1e-3, "weight_decay": wd},
            {"params": oth_nodecay, "lr": 1e-3, "weight_decay": 0.0},
        ])

        print(
            f"[bridgeModelV2] Loss={loss_name}  "
            f"encoder LR=1e-5, other LR=1e-3, WD={wd}  "
            f"fp16={self.use_fp16}"
        )

    def gen_batch_data(self, batched_data: dict) -> dict:
        dict_data = {}

        encoding = self.tokenizer(
            batched_data["sentences_raw"],
            padding        = "max_length",
            truncation     = True,
            max_length     = self.max_length_sen,
            return_tensors = "pt",
        )
        dict_data["input_ids"]      = encoding["input_ids"].to(self.device)
        dict_data["attention_mask"] = encoding["attention_mask"].to(self.device)

        dict_data["sens"]    = self.lang.VariablesFromSentences(
            batched_data["sentences"], True, self.device
        )
        dict_data["len_sen"] = batched_data["length_sen"]

        dict_data["dependency_graph"] = torch.FloatTensor(
            batched_data["dependency_graphs"]
        ).to(self.device)

        dict_data["pragmatic_features"] = torch.FloatTensor(
            batched_data["pragmatic_features"]
        ).to(self.device)

        dict_data["sarcasms"] = torch.LongTensor(
            batched_data["sarcasms"]
        ).to(self.device)

        return dict_data

    def stepTrain(self, batched_data: dict, inference: bool = False):
        """One forward pass (+ backward if not inference).

        Returns:
            (loss_val: float, prob_np: np.ndarray [B, 2])
        """
        train_mode = not inference
        if train_mode:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        b_data  = self.gen_batch_data(batched_data)
        amp_ctx = autocast("cuda") if self.use_fp16 else nullcontext()

        with amp_ctx:
            result = self.model(b_data)
            prob, logits = result[0], result[1]

        loss = self.criterion(logits.float(), b_data["sarcasms"])

        if train_mode:
            if self.use_fp16:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

        return loss.item(), prob.detach().cpu().numpy()

    def predict(self, batched_data: dict):
        self.model.eval()
        with torch.no_grad():
            b_data = self.gen_batch_data(batched_data)
            result = self.model(b_data)
            prob = result[0]
            return prob.argmax(dim=-1).cpu().numpy()
