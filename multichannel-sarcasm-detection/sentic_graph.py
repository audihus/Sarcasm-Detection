# -*- coding: utf-8 -*-

import numpy as np
import stanza
import pickle
import pandas as pd
import os
import urllib.request
from tqdm import tqdm

# Download model Stanza bahasa Indonesia (jika belum ada)
stanza.download('id')

# Inisialisasi Stanza untuk Bahasa Indonesia
nlp = stanza.Pipeline('id', processors='tokenize,mwt,pos,lemma')

def load_sentic_word():
    """
    Download (if necessary) and load InSet Lexicon (Positive & Negative),
    then normalize the values by dividing by 5.0 (so scale is -1.0 to 1.0)
    """
    pos_url = 'https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv'
    neg_url = 'https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv'
    
    pos_path = 'positive.tsv'
    neg_path = 'negative.tsv'
    
    # Auto-download lexicons
    if not os.path.exists(pos_path):
        print("Downloading InSet positive lexicon...")
        urllib.request.urlretrieve(pos_url, pos_path)
    if not os.path.exists(neg_path):
        print("Downloading InSet negative lexicon...")
        urllib.request.urlretrieve(neg_url, neg_path)

    senticNet = {}
    
    # Load positive lexicon
    with open(pos_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            word, weight = line.split('\t')
            try:
                # Normalize to -1.0 to 1.0 scale (skip header row)
                senticNet[word] = float(weight) / 5.0
            except ValueError:
                continue

    # Load negative lexicon
    with open(neg_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            word, weight = line.split('\t')
            try:
                # Normalize to -1.0 to 1.0 scale (skip header row)
                senticNet[word] = float(weight) / 5.0
            except ValueError:
                continue

    return senticNet


def dependency_adj_matrix(text, senticNet):
    document = nlp(text)
    
    # Flatten words from all sentences
    word_list = []
    for sentence in document.sentences:
        for word in sentence.words:
            # Menggunakan lemma, fallback ke teks asli jika tidak ada lemma
            lemma = word.lemma if word.lemma else word.text
            word_list.append(str(lemma).lower())
            
    seq_len = len(word_list)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    
    for i in range(seq_len):
        for j in range(i, seq_len):
            word_i = word_list[i]
            word_j = word_list[j]
            
            if word_i not in senticNet or word_j not in senticNet or word_i == word_j:
                continue
                
            # Logika simetris sentimen: selisih absolut antar nilai kata
            sentic = abs(float(senticNet[word_i] - senticNet[word_j]))
            matrix[i][j] = sentic
            matrix[j][i] = sentic

    return matrix

def process(data_input, text_column, output_filename):
    senticNet = load_sentic_word()
    
    # Support untuk membaca dari path file CSV atau objek DataFrame pandas langsung
    if isinstance(data_input, str):
        print(f"Reading CSV from {data_input}...")
        df = pd.read_csv(data_input)
    elif isinstance(data_input, pd.DataFrame):
        df = data_input.copy()
    else:
        raise ValueError("data_input harus berupa path string (.csv) atau objek Pandas DataFrame")

    if text_column not in df.columns:
        raise ValueError(f"Kolom {text_column} tidak ditemukan di dalam dataset!")

    idx2graph = {}
    
    print(f"Generating Sentic Graphs for {len(df)} records...")
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        text = str(row[text_column]).lower().strip()
        
        # Jika teks kosong, buat matrix 1x1 
        if not text:
            idx2graph[idx] = np.zeros((1, 1)).astype('float32')
            continue
            
        adj_matrix = dependency_adj_matrix(text, senticNet)
        idx2graph[idx] = adj_matrix

    with open(output_filename, 'wb') as fout:
        pickle.dump(idx2graph, fout)
        
    print(f'Done! Sentic graph saved to: {output_filename}')


if __name__ == '__main__':
    # Contoh Penggunaan:
    # 
    # Menggunakan DataFrame langsung:
    # df_mock = pd.DataFrame({"text": ["Bagus sekali produk ini", "Pelayanannya sangat lambat dan buruk."]})
    # process(df_mock, text_column="text", output_filename="sample_dataset.sentic")
    #
    # Atau dari file CSV:
    # process('data/train.csv', text_column='tweet', output_filename='train.sentic')
    pass