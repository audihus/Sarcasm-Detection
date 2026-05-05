#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exploratory Data Analysis (EDA) untuk Deteksi Sarkasme Bahasa Indonesia (Reddit vs Twitter)
Author: AI Data Scientist / NLP Researcher
Description: Script ini menghasilkan metrik dan visualisasi cross-domain untuk dataset Reddit dan Twitter.
Requirements: pandas, seaborn, matplotlib, scikit-learn, stanza, transformers, umap-learn, torch, matplotlib-venn, emoji, scipy, nltk, datasets
"""

# ==========================================
# PERSIAPAN GOOGLE COLAB
# ==========================================
"""
# JALANKAN BLOK INI DI SEL PERTAMA GOOGLE COLAB ANDA:
!pip install pandas seaborn matplotlib scikit-learn stanza transformers umap-learn torch matplotlib-venn emoji scipy nltk datasets

# MOUNT GOOGLE DRIVE UNTUK MENGAKSES DATASET:
from google.colab import drive
drive.mount('/content/drive')

# CONTOH PENGGUNAAN PATH DI COLAB NANTINYA:
# python eda_sarcasm.py --reddit "/content/drive/MyDrive/.../reddit_indonesia_sarcastic" ...
"""

import os
import re
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from matplotlib_venn import venn2
import torch
from transformers import AutoTokenizer, AutoModel
import umap.umap_ as umap
import emoji
import stanza
import argparse
from datasets import load_from_disk
from scipy.stats import mannwhitneyu
from tqdm import tqdm
import nltk
import warnings

# Suppress annoying warnings
warnings.filterwarnings('ignore')

# ==========================================
# SETUP & CONFIGURATION
# ==========================================
# Set seaborn formatting for high-quality academic paper figures
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ==========================================
# CONSTANTS & LEXICONS
# ==========================================
print("Mempersiapkan Lexicon dan Stopwords...")
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# 1. Stopwords menggunakan NLTK + Custom Word
try:
    ID_STOPWORDS = set(stopwords.words('indonesian')).union({'yg', 'aja', 'kalo', 'aku', 'nya', 'sih', 'deh', 'dong', 'kok', 'di', 'ke', 'dari'})
except:
    ID_STOPWORDS = set([
        "yang", "di", "ke", "dari", "pada", "dalam", "untuk", "dengan", "dan", "atau", "ini", "itu", 
        "juga", "sudah", "saya", "kamu", "dia", "mereka", "kita", "kami", "akan", "bisa", "ada", 
        "tidak", "tak", "bukan", "belum", "sangat", "paling", "nya", "yg", "aja", "kalo", "aku"
    ])

# 2. Sentimen Lexicon menggunakan InSet (Indonesian Sentiment Lexicon) oleh Fajri Koto
def load_inset_lexicons():
    os.makedirs('data/lexicon', exist_ok=True)
    pos_path = 'data/lexicon/positive.tsv'
    neg_path = 'data/lexicon/negative.tsv'
    
    pos_url = "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv"
    neg_url = "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv"
    
    if not os.path.exists(pos_path):
        print("Mengunduh InSet Positive Lexicon...")
        urllib.request.urlretrieve(pos_url, pos_path)
    if not os.path.exists(neg_path):
        print("Mengunduh InSet Negative Lexicon...")
        urllib.request.urlretrieve(neg_url, neg_path)
        
    pos_df = pd.read_csv(pos_path, sep='\t')
    neg_df = pd.read_csv(neg_path, sep='\t')
    
    # Ambil kolom kata yang memiliki bobot sentimen kuat
    pos_words = set(pos_df[pos_df['weight'] > 2]['word'].dropna().tolist()) if 'weight' in pos_df.columns else set(pos_df['word'].dropna().tolist())
    neg_words = set(neg_df[neg_df['weight'] < -2]['word'].dropna().tolist()) if 'weight' in neg_df.columns else set(neg_df['word'].dropna().tolist())
    
    return pos_words, neg_words

try:
    ID_POS_LEXICON, ID_NEG_LEXICON = load_inset_lexicons()
    print(f"Lexicon InSet berhasil dimuat: {len(ID_POS_LEXICON)} kata positif, {len(ID_NEG_LEXICON)} kata negatif.")
except Exception as e:
    print(f"Gagal mengunduh InSet lexicon ({e}), menggunakan fallback lexicon manual.")
    ID_POS_LEXICON = set(["bagus", "keren", "mantap", "hebat", "pintar", "cerdas", "baik", "cantik", "indah", "luar biasa", "sempurna", "terbaik", "ganteng", "cakep", "lucu", "gemas", "kreatif", "inovatif", "jenius", "suka", "cinta", "sayang", "asik", "keren"])
    ID_NEG_LEXICON = set(["jelek", "buruk", "bodoh", "tolol", "goblok", "bego", "idiot", "hancur", "parah", "sampah", "najis", "jijik", "benci", "marah", "kesal", "kecewa", "payah", "gagal", "lemah", "malas", "sial", "sialan", "bangsat", "anjing", "babi"])

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def load_data(reddit_path, twitter_path):
    print("Loading datasets dari format HuggingFace...")
    try:
        hf_reddit = load_from_disk(reddit_path)
        df_reddit_splits = [hf_reddit[split].to_pandas() for split in hf_reddit.keys()]
        df_reddit = pd.concat(df_reddit_splits, ignore_index=True)
        print(f"Reddit dataset loaded: {df_reddit.shape}")
    except Exception as e:
        print(f"Error loading Reddit dataset: {e}")
        df_reddit = None
        
    try:
        hf_twitter = load_from_disk(twitter_path)
        df_twitter_splits = [hf_twitter[split].to_pandas() for split in hf_twitter.keys()]
        df_twitter = pd.concat(df_twitter_splits, ignore_index=True)
        print(f"Twitter dataset loaded: {df_twitter.shape}")
    except Exception as e:
        print(f"Error loading Twitter dataset: {e}")
        df_twitter = None

    if df_reddit is None or df_twitter is None:
        print("Satu atau lebih dataset gagal di-load. Menghasilkan dummy data untuk keperluan demo pipeline...")
        df_reddit, df_twitter = generate_dummy_data()

    print("Mengecek missing values...")
    df_reddit.dropna(subset=['text', 'label'], inplace=True)
    df_twitter.dropna(subset=['tweet', 'label'], inplace=True)

    df_reddit = df_reddit.rename(columns={'text': 'clean_text'})
    df_twitter = df_twitter.rename(columns={'tweet': 'clean_text'})
    
    return df_reddit, df_twitter

def generate_dummy_data():
    reddit_data = {
        'text': ['Wah bagus banget ya pemerintah kita!', 'Hari ini hujan lagi.', 'Suka banget lihat macet Jakarta.', 'Saya sedang makan nasi goreng.'],
        'score': [-5, 10, -2, 15],
        'label': [1, 0, 1, 0]
    }
    twitter_data = {
        'tweet': ['Keren banget deh kelakuan lu wkwk', 'Selamat pagi dunia!', 'Pintar sekali ya sampai nilai 0', 'Ayo semangat belajar!'],
        'label': [1, 0, 1, 0]
    }
    return pd.DataFrame(reddit_data), pd.DataFrame(twitter_data)

# ==========================================
# A. CROSS-DOMAIN ANALYSIS
# ==========================================

def analyze_text_length(df_reddit, df_twitter, output_dir):
    print("Menganalisis Distribusi Panjang Teks...")
    df_reddit['word_count'] = df_reddit['clean_text'].apply(lambda x: len(str(x).split()))
    df_twitter['word_count'] = df_twitter['clean_text'].apply(lambda x: len(str(x).split()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(data=df_reddit, x='word_count', hue='label', bins=30, kde=True, ax=axes[0], palette='Set2')
    axes[0].set_title("Distribusi Panjang Kata - Reddit")
    axes[0].set_xlabel("Jumlah Kata")
    axes[0].set_ylabel("Frekuensi")
    
    sns.histplot(data=df_twitter, x='word_count', hue='label', bins=30, kde=True, ax=axes[1], palette='Set2')
    axes[1].set_title("Distribusi Panjang Kata - Twitter")
    axes[1].set_xlabel("Jumlah Kata")
    axes[1].set_ylabel("Frekuensi")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "text_length_distribution.png"))
    plt.close()

def extract_top_words(texts, top_n=20, ngram_range=(1, 2)):
    # Menggunakan TfidfVectorizer untuk menekan bobot kata umum
    vec = TfidfVectorizer(stop_words=list(ID_STOPWORDS), max_features=1500, ngram_range=ngram_range)
    bag_of_words = vec.fit_transform(texts)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return [word for word, freq in words_freq[:top_n]]

def analyze_lexical_overlap(df_reddit, df_twitter, output_dir):
    print("Menganalisis Irisan Kosakata & N-Grams (Lexical Overlap)...")
    reddit_sarcasm = df_reddit[df_reddit['label'] == 1]['clean_text'].astype(str)
    twitter_sarcasm = df_twitter[df_twitter['label'] == 1]['clean_text'].astype(str)
    
    top_reddit = set(extract_top_words(reddit_sarcasm, top_n=100, ngram_range=(1, 2)))
    top_twitter = set(extract_top_words(twitter_sarcasm, top_n=100, ngram_range=(1, 2)))
    
    if not top_reddit and not top_twitter:
        print("Data tidak cukup untuk Lexical Overlap")
        return
        
    plt.figure(figsize=(8, 6))
    venn_out = venn2([top_reddit, top_twitter], ('Reddit Sarcasm', 'Twitter Sarcasm'))
    plt.title("Lexical Overlap Top-100 Words/Phrases Sarcasm (Reddit vs Twitter)")
    plt.savefig(os.path.join(output_dir, "lexical_overlap_venn.png"))
    plt.close()

    reddit_exclusive = list(top_reddit - top_twitter)[:10]
    twitter_exclusive = list(top_twitter - top_reddit)[:10]
    
    print(f"Kata/Frasa Eksklusif Sering Muncul di Reddit Sarcasm:\n-> {reddit_exclusive}")
    print(f"Kata/Frasa Eksklusif Sering Muncul di Twitter Sarcasm:\n-> {twitter_exclusive}")

# ==========================================
# B. LINGUISTIC & PRAGMATIC ANALYSIS
# ==========================================

def analyze_particles_slang(df, dataset_name, output_dir):
    print(f"Menganalisis Partikel Bahasa & Slang ({dataset_name})...")
    particles = ['sih', 'kok', 'deh', 'dong', 'cie', 'wkwk', 'idih']
    
    results = []
    for label in [0, 1]:
        subset = df[df['label'] == label]['clean_text'].astype(str)
        text_joined = " ".join(subset).lower()
        # Perbaikan regex: Menggunakan exact match boundaries \b agar tidak mencocokkan kata yang berawalan partikel
        counts = {p: len(re.findall(r'\b' + p + r'\b', text_joined)) for p in particles}
        
        total_docs = len(subset)
        if total_docs > 0:
            counts_normalized = {p: count/total_docs for p, count in counts.items()}
        else:
            counts_normalized = counts
            
        counts_normalized['label'] = 'Sarkas' if label == 1 else 'Non-Sarkas'
        results.append(counts_normalized)
        
    df_res = pd.DataFrame(results).melt(id_vars='label', var_name='Particle', value_name='Average Frequency per Doc')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_res, x='Particle', y='Average Frequency per Doc', hue='label', palette='Set1')
    plt.title(f"Frekuensi Partikel Bahasa & Slang Khas - {dataset_name}")
    plt.savefig(os.path.join(output_dir, f"particles_slang_{dataset_name.lower()}.png"))
    plt.close()

def analyze_syntax(df, dataset_name, output_dir):
    print(f"Menganalisis Pola Sintaksis (POS Tagging) - {dataset_name}...")
    try:
        stanza.download('id', processors='tokenize,pos', verbose=False)
        nlp = stanza.Pipeline('id', processors='tokenize,pos', use_gpu=True, verbose=False)
    except Exception as e:
        print(f"Gagal memuat Stanza Pipeline: {e}")
        return
        
    adj_ratios = []
    adv_adj_counts = []
    
    # Sampling for performance reasons, Stanza is very slow on large datasets
    sample_size = min(df['label'].value_counts().min(), 300)
    if sample_size < 2: 
        print("Data terlalu sedikit untuk POS Tagging")
        return
    
    sample_df = df.groupby('label').apply(lambda x: x.sample(sample_size, random_state=42)).reset_index(drop=True)
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc=f"POS Tagging {dataset_name}"):
        doc = nlp(str(row['clean_text']))
        num_adj = 0
        num_tokens = 0
        adv_adj_pattern = 0
        
        for sentence in doc.sentences:
            words = sentence.words
            num_tokens += len(words)
            for i in range(len(words)):
                if words[i].upos == 'ADJ':
                    num_adj += 1
                if i < len(words) - 1 and words[i].upos == 'ADV' and words[i+1].upos == 'ADJ':
                    adv_adj_pattern += 1
                    
        adj_ratio = num_adj / num_tokens if num_tokens > 0 else 0
        adj_ratios.append(adj_ratio)
        adv_adj_counts.append(adv_adj_pattern)
        
    sample_df['adj_ratio'] = adj_ratios
    sample_df['adv_adj_count'] = adv_adj_counts
    
    # Statistical test
    sarcasm_ratios = sample_df[sample_df['label'] == 1]['adj_ratio']
    non_sarcasm_ratios = sample_df[sample_df['label'] == 0]['adj_ratio']
    stat, p = mannwhitneyu(sarcasm_ratios, non_sarcasm_ratios)
    
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(data=sample_df, x='label', y='adj_ratio', palette='Pastel1')
    plt.title(f"Rasio Kemunculan Kata Sifat (Adjective) - {dataset_name}\nMann-Whitney p-value: {p:.4f}")
    plt.xticks([0, 1], ['Non-Sarkas', 'Sarkas'])
    plt.ylabel("Adjective Ratio")
    plt.savefig(os.path.join(output_dir, f"syntax_adj_ratio_{dataset_name.lower()}.png"))
    plt.close()

def analyze_polarity_clash(df, dataset_name):
    print(f"Menganalisis Polarity Clash ({dataset_name})...")
    def get_pos_score(text):
        words = set(str(text).lower().split())
        return len(words.intersection(ID_POS_LEXICON))
        
    df['pos_lexicon_score'] = df['clean_text'].apply(get_pos_score)
    clash_samples = df[(df['label'] == 1) & (df['pos_lexicon_score'] >= 2)]
    
    print(f"\nContoh Polarity Clash di {dataset_name} (Label: Sarkas, Sentimen Lexicon: Positif):")
    for text in clash_samples['clean_text'].head(5):
        print(f"-> {text}")

def analyze_punctuation_emoji(df, dataset_name, output_dir):
    print(f"Menganalisis Tanda Baca & Emoji ({dataset_name})...")
    
    df['exclamation_count'] = df['clean_text'].apply(lambda x: str(x).count('!'))
    df['question_count'] = df['clean_text'].apply(lambda x: str(x).count('?'))
    df['ellipse_count'] = df['clean_text'].apply(lambda x: len(re.findall(r'\.\.\.', str(x))))
    df['emoji_count'] = df['clean_text'].apply(lambda x: emoji.emoji_count(str(x)))
    
    metrics = ['exclamation_count', 'question_count', 'ellipse_count', 'emoji_count']
    
    print(f"\n[Statistik Signifikansi (p-value) {dataset_name}]")
    sarcasm_df = df[df['label'] == 1]
    nonsarcasm_df = df[df['label'] == 0]
    for metric in metrics:
        stat, p = mannwhitneyu(sarcasm_df[metric].dropna(), nonsarcasm_df[metric].dropna())
        sig = "Signifikan" if p < 0.05 else "Tidak Signifikan"
        print(f"- {metric}: p-value = {p:.4e} ({sig})")
    
    agg_df = df.groupby('label')[metrics].mean().reset_index()
    agg_df['label'] = agg_df['label'].map({0: 'Non-Sarkas', 1: 'Sarkas'})
    agg_df_melted = agg_df.melt(id_vars='label', var_name='Feature', value_name='Average Count')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=agg_df_melted, x='Feature', y='Average Count', hue='label', palette='Set2')
    plt.title(f"Rata-rata Penggunaan Tanda Baca & Emoji - {dataset_name}")
    plt.xticks(rotation=15)
    plt.savefig(os.path.join(output_dir, f"punctuation_emoji_{dataset_name.lower()}.png"))
    plt.close()

# ==========================================
# C. EKSPLORASI RUANG SEMANTIK
# ==========================================

def semantic_projection(df, dataset_name, output_dir, model_name="indobenchmark/indobert-lite-base-p1"):
    print(f"Mengekstraksi embedding dan membuat proyeksi UMAP ({dataset_name})...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Gagal meload IndoBERT: {e}")
        return

    model.eval()
    
    # Meningkatkan batas sampling untuk model kompak
    sample_size = min(df['label'].value_counts().min(), 2000)
    if sample_size < 2: 
        print("Data terlalu sedikit untuk Semantic Projection")
        return
    
    sample_df = df.groupby('label').apply(lambda x: x.sample(sample_size, random_state=42)).reset_index(drop=True)
    texts = sample_df['clean_text'].tolist()
    labels = sample_df['label'].tolist()
    
    embeddings = []
    batch_size = 32
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"IndoBERT Extract {dataset_name}"):
            batch_texts = [str(t) for t in texts[i:i+batch_size]]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(cls_embeddings)
            
    embeddings = np.array(embeddings)
    
    print(f"Menjalankan dimensionality reduction dengan UMAP ({dataset_name})...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    proj_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=proj_2d[:, 0], y=proj_2d[:, 1], hue=labels, palette=['blue', 'red'], alpha=0.7)
    plt.title(f"Proyeksi UMAP Ruang Semantik ({model_name}) - {dataset_name}")
    
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, title='Label', labels=['Non-Sarkas', 'Sarkas'])
    
    plt.savefig(os.path.join(output_dir, f"semantic_umap_{dataset_name.lower()}.png"))
    plt.close()

# ==========================================
# D. ANALISIS METADATA SPESIFIK REDDIT
# ==========================================

def analyze_reddit_metadata(df_reddit, output_dir):
    print("Menganalisis Korelasi Skor dan Sarkasme (Reddit)...")
    if 'score' not in df_reddit.columns:
        print("Kolom 'score' tidak ditemukan di dataset Reddit. Melewati analisis metadata spesifik Reddit.")
        return
        
    bins = [-np.inf, 0, 10, np.inf]
    labels = ['Negatif (< 0)', 'Rendah (0-10)', 'Tinggi (> 10)']
    df_reddit['score_bin'] = pd.cut(df_reddit['score'], bins=bins, labels=labels)
    
    sarcasm_pct = df_reddit.groupby('score_bin', observed=False)['label'].mean() * 100
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=sarcasm_pct.index, y=sarcasm_pct.values, palette='viridis')
    plt.title("Persentase Komentar Sarkas Berdasarkan Kelompok Skor Reddit")
    plt.xlabel("Kelompok Skor (Upvotes/Downvotes)")
    plt.ylabel("Persentase Sarkasme (%)")
    
    for i, v in enumerate(sarcasm_pct.values):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
        
    plt.ylim(0, max(sarcasm_pct.values) + 15)
    plt.savefig(os.path.join(output_dir, "reddit_score_sarcasm_correlation.png"))
    plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA Deteksi Sarkasme Bahasa Indonesia")
    parser.add_argument('--reddit', type=str, default='data/reddit_indonesia_sarcastic', help='Path ke direktori dataset HF Reddit')
    parser.add_argument('--twitter', type=str, default='data/twitter_indonesia_sarcastic', help='Path ke direktori dataset HF Twitter')
    parser.add_argument('--outdir', type=str, default='results/eda_plots', help='Directory untuk menyimpan output plot')
    
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)

    print("=== Memulai Exploratory Data Analysis (EDA) Sarkasme Bahasa Indonesia ===")
    df_red, df_tw = load_data(args.reddit, args.twitter)
    
    print("\n--- A. Analisis Lintas Domain ---")
    analyze_text_length(df_red, df_tw, args.outdir)
    analyze_lexical_overlap(df_red, df_tw, args.outdir)
    
    print("\n--- B. Analisis Linguistik dan Pragmatik ---")
    for df, name in [(df_red, 'Reddit'), (df_tw, 'Twitter')]:
        analyze_particles_slang(df, name, args.outdir)
        analyze_syntax(df, name, args.outdir)
        analyze_punctuation_emoji(df, name, args.outdir)
        
    print("\n[Menganalisis Polarity Clash]")
    analyze_polarity_clash(df_red, "Reddit")
    analyze_polarity_clash(df_tw, "Twitter")
    
    print("\n--- C. Eksplorasi Ruang Semantik ---")
    semantic_projection(df_red, "Reddit", args.outdir)
    semantic_projection(df_tw, "Twitter", args.outdir)
    
    print("\n--- D. Analisis Metadata Spesifik Reddit ---")
    analyze_reddit_metadata(df_red, args.outdir)
    
    print(f"\n=== EDA Selesai! Semua visualisasi disimpan dalam folder '{args.outdir}' ===")
