# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

This workspace contains two independent research projects for Indonesian sarcasm detection:

```
multi_channel_method/
├── id_sarcasm/                          # Benchmarking project (IEEE Access 2024)
│   └── CLAUDE.md                        # Detailed guidance for id_sarcasm
└── multichannel-sarcasm-detection/      # Multi-channel deep learning model
```

---

## `multichannel-sarcasm-detection/` — Multi-Channel Deep Learning Model

A PyTorch implementation that detects sarcasm by fusing a BERT encoder with a graph-based encoder (ADGCN) over dependency and sentiment graphs.

### Pipeline

```bash
# 1. Dependency graph construction (uses Stanza Indonesian NLP pipeline)
python dependency_graph.py

# 2. Sentiment graph construction (uses SentiWordNet_3.0.0.txt)
python sentic_graph.py

# 3. Train
bash train.sh

# 4. Evaluate a saved checkpoint
python evaluate.py --checkpoint checkpoints/best_model.pt --dataset IAC1

# 5. Inference on new data
python predict_sarcasm.py
```

### Architecture

The core model (`Model.py: dualModel`) has two parallel encoders whose outputs are concatenated and fed to a dense classifier:

1. **IndoBERT-base encoder** (`indobenchmark/indobert-base-p1`) — `pooler_output` (pretrained Linear(768→768)+tanh on CLS) → 768-dim vector.
2. **ADGCN** (`GarphModel.py: ADGCN`) — BiLSTM + 5-layer GCN operating on two adjacency matrices (dependency graph + sentic graph) → `dim_hidden * 2` vector.
3. **Fusion**: `concat[bert_rep, adgcn_rep]` → `Dropout(0.1) → Linear(768+dim_adgcn, 2)` (1-layer head matching HF `AutoModelForSequenceClassification` style).

Training uses **Focal Loss** (γ=2, α=0.75) with differential learning rates: IndoBERT-base at 1e-5, ADGCN/Dense at 1e-3 (AdamW, gradient clipping 1.0). Weight decay scope follows HF Trainer convention (skip `bias` and `LayerNorm.weight`). fp16 mixed precision (autocast + GradScaler) is enabled by default when CUDA is available.

### Key Files

| File | Role |
|------|------|
| `main.py` | Entry point; `argparse` config, training loop, TensorBoard logging |
| `bridgeModel.py` | Wraps `dualModel`; handles tokenization, batch prep, FocalLoss, optimizer |
| `Model.py` | `dualModel` — fuses BERT CLS + ADGCN output |
| `GarphModel.py` | `ADGCN` — BiLSTM + 5-layer GCN (note: filename is a typo, intentional) |
| `dataUtils.py` | `DataManager` — loads `.txt`/`.graph.new`/`.sentic` files, builds vocab & embedding matrix |
| `dependency_graph.py` | Builds syntactic adjacency matrices via Stanza Indonesian depparse |
| `sentic_graph.py` | Builds semantic adjacency matrices via SentiWordNet scores |
| `evaluation.py` | Computes Acc, F1-micro/macro, Precision, Recall, AUC |

### Data Format

Each dataset split lives under `<DATASET>/spacy/` and requires three aligned files:
- `train.txt` / `test.txt` — alternating lines: sentence (even) then label `0`/`1` (odd), or JSON per line
- `train.txt.graph.new` / `test.txt.graph.new` — pickled `dict[int → np.ndarray]` adjacency matrices
- `train.txt.sentic` / `test.txt.sentic` — pickled `dict[int → np.ndarray]` sentiment graph matrices

Word vectors are loaded from `<data_dir>/vectors.glove.300d.txt` (preferred) or `./senti/glove.840B.300d.txt`. An embedding matrix cache (`<dim>_<dataset>_embedding_matrix.pkl`) is written to `data_dir` on first run.

### Key `train.sh` Parameters

| Argument | Default | Notes |
|----------|---------|-------|
| `--data_dir` | `./IAC2/spacy/` | Path to preprocessed dataset |
| `--name_dataset` | dataset name | Used for log filenames |
| `--voc_size` | 30000 | Vocabulary size cap |
| `--batch_size` | 128 | Batch size |
| `--n_layers` | 3 | GCN layers |
| `--iter_num` | 32×150 | Total training steps |
| `--per_checkpoint` | 64 | Steps between eval/log |
| `--save_model` | 0 | Set to 1 to save checkpoints |
| `--breakpoint` | -1 | Resume from checkpoint index |

Logs are written to `logs/<dataset>_<model>_<timestamp>.log`. TensorBoard summaries go to `summary/`.

### Important Implementation Notes

- `GarphModel.py` has a deliberate filename typo (Graph → Garph); do not rename it — imports depend on the exact name.
- `dataUtils.py: DataManager.__init__` contains a hardcoded path `/root/autodl-tmp/DCNet/bert_pretrain` for the BERT tokenizer — update this for local environments.
- `dependency_graph.py` uses Stanza Indonesian pipeline (`stanza.download('id')` runs on import); ensure network access or a pre-downloaded model on first run.
- The `--t_sne` flag enables returning intermediate representations (`bert_rep`, `adgcn_rep`, `dense_input`) for visualization; only use with `--predict 1`.

---

## `id_sarcasm/` — Indonesian Sarcasm Benchmark

See `id_sarcasm/CLAUDE.md` for full details. This sub-project benchmarks classical ML, fine-tuned transformers (IndoBERT, mBERT, XLM-R), and zero-shot LLMs (BLOOMZ, mT0) on Indonesian sarcasm datasets from Reddit and Twitter.
