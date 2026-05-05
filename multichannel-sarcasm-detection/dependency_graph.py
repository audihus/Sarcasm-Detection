# -*- coding: utf-8 -*-

import numpy as np
import stanza
import pickle
import pandas as pd
from tqdm import tqdm

# Download model Stanza bahasa Indonesia (jika belum ada)
stanza.download('id')

# Inisialisasi Stanza untuk Bahasa Indonesia dengan depparse (Dependency Parsing)
nlp = stanza.Pipeline('id', processors='tokenize,mwt,pos,lemma,depparse')

def dependency_adj_matrix(text):
    document = nlp(text)
    
    # Hitung total panjang kata di seluruh kalimat untuk ukuran matrix
    seq_len = sum([len(sentence.words) for sentence in document.sentences])
    
    # Jika input kosong
    if seq_len == 0:
        return np.zeros((1, 1)).astype('float32')

    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    
    global_offset = 0  # Offset untuk mengakumulasi ID indeks dari kalimat sebelumnya
    
    for sentence in document.sentences:
        for word in sentence.words:
            # Stanza 1-based index (di-reset per sentence), jadi dikurang 1
            idx_saat_ini = word.id - 1 + global_offset
            
            # Diagonal node ke dirinya sendiri
            matrix[idx_saat_ini][idx_saat_ini] = 1
            
            # word.head = 0 menandakan dia adalah Root, dan tidak memiliki induk
            if word.head > 0:
                idx_induk = word.head - 1 + global_offset
                
                # Buat koneksi simetris undirected graph
                matrix[idx_saat_ini][idx_induk] = 1
                matrix[idx_induk][idx_saat_ini] = 1
                
        # Tambahkan jumlah kata di kalimat ini ke global_offset untuk iterasi kalimat selanjutnya
        global_offset += len(sentence.words)

    return matrix

def process(data_input, text_column, output_filename):
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

    print(f"Generating Dependency Graphs for {len(df)} records...")
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        text = str(row[text_column]).lower().strip()
        
        # Jika teks kosong, buat matrix 1x1 
        if not text:
            idx2graph[idx] = np.zeros((1, 1)).astype('float32')
            continue
            
        adj_matrix = dependency_adj_matrix(text)
        idx2graph[idx] = adj_matrix
        
    with open(output_filename, 'wb') as fout:
        pickle.dump(idx2graph, fout)
        
    print(f'Done! Dependency graph saved to: {output_filename}')

if __name__ == '__main__':
    # Contoh Penggunaan:
    # 
    # Menggunakan DataFrame langsung:
    # df_mock = pd.DataFrame({"text": ["Bagus sekali produk ini", "Pelayanannya sangat lambat dan buruk."]})
    # process(df_mock, text_column="text", output_filename="sample_dataset.graph")
    #
    # Atau dari file CSV:
    # process('data/train.csv', text_column='tweet', output_filename='train.graph')
    pass