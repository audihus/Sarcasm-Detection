from transformers import AutoTokenizer
import torch

# Pakai base tokenizer + register special tokens manual
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
tokenizer.add_special_tokens({"additional_special_tokens": ["[HAS_CTX]", "[NO_CTX]", "[CTX]", "[TGT]"]})

ctx_id = tokenizer.convert_tokens_to_ids("[CTX]")
tgt_id = tokenizer.convert_tokens_to_ids("[TGT]")
sep_id = tokenizer.sep_token_id

# ===== Test sampel has-parent =====
text_has = "[HAS_CTX] [CTX] saya suka hujan [TGT] iya bagus sekali"
ids_has = tokenizer.encode(text_has, return_tensors="pt")
print("=" * 60)
print("HAS_CTX sample:")
print("  Text:", text_has)
print("  Tokens:", tokenizer.convert_ids_to_tokens(ids_has[0].tolist()))
print("  IDs:", ids_has[0].tolist())

ctx_pos = (ids_has[0] == ctx_id).nonzero(as_tuple=True)[0]
tgt_pos = (ids_has[0] == tgt_id).nonzero(as_tuple=True)[0]
sep_pos = (ids_has[0] == sep_id).nonzero(as_tuple=True)[0]
print(f"  ctx_pos={ctx_pos.tolist()}, tgt_pos={tgt_pos.tolist()}, sep_pos={sep_pos.tolist()}")

idx_ctx = ctx_pos[0].item()
idx_tgt = tgt_pos[0].item()
idx_sep = sep_pos[0].item()
parent_tokens = tokenizer.convert_ids_to_tokens(ids_has[0][idx_ctx+1:idx_tgt].tolist())
comment_tokens = tokenizer.convert_ids_to_tokens(ids_has[0][idx_tgt+1:idx_sep].tolist())
print(f"  Parent span ({idx_ctx+1}:{idx_tgt}): {parent_tokens}")
print(f"  Comment span ({idx_tgt+1}:{idx_sep}): {comment_tokens}")

# ===== Test sampel no-parent =====
print("=" * 60)
text_no = "[NO_CTX] [TGT] iya bagus sekali"
ids_no = tokenizer.encode(text_no, return_tensors="pt")
print("NO_CTX sample:")
print("  Text:", text_no)
print("  Tokens:", tokenizer.convert_ids_to_tokens(ids_no[0].tolist()))

ctx_pos_no = (ids_no[0] == ctx_id).nonzero(as_tuple=True)[0]
tgt_pos_no = (ids_no[0] == tgt_id).nonzero(as_tuple=True)[0]
sep_pos_no = (ids_no[0] == sep_id).nonzero(as_tuple=True)[0]
print(f"  ctx_pos (harus empty): {ctx_pos_no.tolist()}")
print(f"  tgt_pos={tgt_pos_no.tolist()}, sep_pos={sep_pos_no.tolist()}")

idx_tgt_no = tgt_pos_no[0].item()
idx_sep_no = sep_pos_no[0].item()
comment_tokens_no = tokenizer.convert_ids_to_tokens(ids_no[0][idx_tgt_no+1:idx_sep_no].tolist())
print(f"  Comment span ({idx_tgt_no+1}:{idx_sep_no}): {comment_tokens_no}")