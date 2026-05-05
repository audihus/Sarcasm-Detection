# Multi-Channel Sarcasm Detection

This repository provides the official implementation of the paper:

**"A Multi-Channel Sarcasm Detection Model Integrating Syntax and Semantics"**

## 🏃 Running the Model 

You can follow the full pipeline with the commands below.  
Just copy and paste the block to run all steps in order:

```bash
# 1️⃣ Data Preprocessing
python preprocess.py 

# 2️⃣ Dependency Graph Construction
python dependency_graph.py 

# 3️⃣ Sentiment Graph Construction
python sentic_graph.py 

# 4️⃣ Train the Model
bash train.sh

# 5️⃣ Evaluate the Model
python evaluate.py --checkpoint checkpoints/best_model.pt --dataset IAC1


## ⚙️ Requirements
- Python 3.8+
- PyTorch >= 1.10
- Transformers
- NLTK / SpaCy (for POS tagging)
- SenticNet (for sentiment word detection)





