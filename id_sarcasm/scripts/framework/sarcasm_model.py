"""
SarcasmModel: IndoBERT + main head (sarkas) + aux head (incongruity clash).

Forward mengembalikan (main_logits, aux_logits). Pooling mode dikontrol lewat
argumen `pooling` saat konstruksi: "cls" (default) atau "mean".
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class SarcasmModel(nn.Module):
    def __init__(self, name: str, n_tokens: int, pooling: str = "cls"):
        """
        Args:
            name:     HuggingFace model ID, mis. "indobenchmark/indobert-base-p1"
            n_tokens: len(tokenizer) setelah add_special_tokens — untuk resize embedding
            pooling:  "cls" pakai pooler_output; "mean" pakai mean atas token non-padding
        """
        super().__init__()
        assert pooling in ("cls", "mean"), f"pooling harus 'cls' atau 'mean', dapat: {pooling}"
        self.pooling = pooling
        self.enc = AutoModel.from_pretrained(name)
        self.enc.resize_token_embeddings(n_tokens)
        h = self.enc.config.hidden_size       # 768 untuk indobert-base
        self.main = nn.Linear(h, 2)
        self.aux  = nn.Linear(h, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == "cls":
            pooled = out.pooler_output                        # (B, 768)
        else:
            # mean-pool atas token yang bukan padding
            mask = attention_mask.unsqueeze(-1).float()       # (B, L, 1)
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.main(pooled), self.aux(pooled)
