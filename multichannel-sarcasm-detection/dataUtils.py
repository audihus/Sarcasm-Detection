# -*- coding: utf-8 -*-

import logging
from tqdm import tqdm
import json
import numpy as np
import re
from nltk import word_tokenize
import spacy
import os
import pickle
from collections import Counter
from transformers import BertTokenizer

nlp = spacy.load('en_core_web_sm')

def remove_url(text):
    results = re.compile(r'https://[a-zA-Z0-9.?/&=:]*', re.S)
    text = results.sub("", text)
    results = re.compile(r'http://[a-zA-Z0-9.?/&=:]*', re.S)
    text = results.sub("", text)
    return text


# 词向量加载函数
def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                print(f'WARNING: 词向量 {tokens[0]} 加载异常，已跳过')
    return word_vec





# 嵌入矩阵构建函数（带缓存）
def build_embedding_matrix(word2idx, embed_dim, dataset_type, data_dir):
    embedding_matrix_file = os.path.join(data_dir, f'{embed_dim}_{dataset_type}_embedding_matrix.pkl')
    
    if os.path.exists(embedding_matrix_file):
        print(f'加载缓存的嵌入矩阵：{embedding_matrix_file}')
        return pickle.load(open(embedding_matrix_file, 'rb'))
    
    print(f'构建嵌入矩阵：{embedding_matrix_file}')
    embedding_matrix = np.zeros((len(word2idx), embed_dim))
    embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
    
    # 词向量路径兼容
    vec_paths = [
        os.path.join(data_dir, 'vectors.glove.300d.txt'),  # 优先使用当前数据集目录下的词向量
        './senti/glove.840B.300d.txt'                      # 备选路径
    ]
    
    word_vec = {}
    for path in vec_paths:
        if os.path.exists(path):
            word_vec = load_word_vec(path, word2idx)
            break
    
    for word, i in word2idx.items():
        vec = word_vec.get(word)
        if vec is not None:
            if vec.shape[0] == embed_dim:
                embedding_matrix[i] = vec
            elif vec.shape[0] > embed_dim:
                embedding_matrix[i] = vec[:embed_dim]  # 截断
            else:
                embedding_matrix[i, :vec.shape[0]] = vec  # 补零
    
    pickle.dump(embedding_matrix, open(embedding_matrix_file, 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, tokenizer=None, max_seq_len=128, word2idx=None):
        self.bert_tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        if word2idx is None:
            self.word2idx = {'<pad>': 0, 'UNK': 1}
            self.idx2word = {0: '<pad>', 1: 'UNK'}
            self.idx = 2
        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    def text_to_sequence(self, text):
        encoded = self.bert_tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            padding=False,
            max_length=self.max_seq_len,
            return_attention_mask=True
        )
        return encoded['input_ids']

    def fit_on_text(self, text_list):
        for text in text_list:
            text = text.lower().strip()
            words = [str(token) for token in nlp(text)]
            for word in words:
                if word not in self.word2idx:
                    self.word2idx[word] = self.idx
                    self.idx2word[self.idx] = word
                    self.idx += 1


class Dataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DatesetReader:
    @staticmethod
    def get_file_paths(data_dir, data_type):
        """根据数据目录和类型动态生成路径"""
        return os.path.join(data_dir, f"{data_type}.txt")
    
    @staticmethod
    def get_graph_paths(data_dir, base_filename):
        """获取依赖图和情感图的路径（基于原始文件名）"""
        # 保留原始文件名的扩展名部分（如train.txt → train.txt.graph.new）
        graph_path = os.path.join(data_dir, f"{base_filename}.graph.new")
        sentic_path = os.path.join(data_dir, f"{base_filename}.sentic")
        return graph_path, sentic_path

    @staticmethod
    def read_text_files(fnames):
        text = []
        for fname in fnames:
            with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            for i in range(0, len(lines), 2):  # 文本行（偶数行）
                text.append(lines[i].lower().strip())
        return text


class DataManager(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        np.random.seed(FLAGS.seed)
        self.data_dir = FLAGS.data_dir
        self.dataset = FLAGS.name_dataset.lower()  
        
        # 初始化分词器
        self.bert_tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/DCNet/bert_pretrain')
        self.custom_tokenizer = Tokenizer(
            tokenizer=self.bert_tokenizer,
            max_seq_len=FLAGS.max_length_sen
        )

    def load_data(self, path, fname):
        print(f"加载数据：path={path}, fname={fname}")
        
        
        data_type = os.path.splitext(fname)[0]  
        base_filename = fname  
        
        # 使用DatesetReader获取路径（基于FLAGS.data_dir）
        data_path = DatesetReader.get_file_paths(self.data_dir, data_type)
        graph_path, sentic_path = DatesetReader.get_graph_paths(self.data_dir, base_filename)
        
        # 检查文件存在性
        for p in [data_path, graph_path, sentic_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"文件不存在：{p}")
        
        print(f"数据集: {self.dataset}")
        print(f"原始数据路径: {data_path}")
        print(f"依赖图路径: {graph_path}")
        print(f"情感图路径: {sentic_path}")

        # 加载图数据
        with open(graph_path, 'rb') as f:
            idx2graph = pickle.load(f)
        with open(sentic_path, 'rb') as f:
            idx2sentic = pickle.load(f)

        # 读取原始数据
        lines = []
        with open(data_path, 'r', encoding='utf-8') as f:
            # 检测文件格式
            first_line = f.readline().strip()
            f.seek(0)  # 回到文件开头
            
            if first_line.startswith('{'):  # JSON格式
                lines = [json.loads(line.strip()) for line in f.readlines() if line.strip()]
            else:  # 交替文本+标签格式
                content_lines = []
                label_lines = []
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    if i % 2 == 0:
                        content_lines.append(line)
                    else:
                        label_lines.append(line)
                min_len = min(len(content_lines), len(label_lines))
                for i in range(min_len):
                    lines.append({
                        'content': content_lines[i],
                        'label': label_lines[i],
                        'pos': [], 'neg': [], 'senti': [], 'nonsenti': []
                    })

        # 打印数据统计信息
        print(f"原始数据样本数: {len(lines)}")
        print(f"依赖图数据样本数: {len(idx2graph)}")
        print(f"情感图数据样本数: {len(idx2sentic)}")
        
        # 确保数据长度匹配
        min_samples = min(len(lines), len(idx2graph), len(idx2sentic))
        if len(lines) > min_samples:
            print(f"警告: 原始数据({len(lines)})多于依赖图数据({min_samples})，将截断原始数据")
            lines = lines[:min_samples]
        if len(idx2graph) > min_samples:
            print(f"警告: 依赖图数据({len(idx2graph)})多于原始数据({min_samples})，将截断依赖图数据")
            idx2graph = {k: idx2graph[k] for k in list(idx2graph.keys())[:min_samples]}
        if len(idx2sentic) > min_samples:
            print(f"警告: 情感图数据({len(idx2sentic)})多于原始数据({min_samples})，将截断情感图数据")
            idx2sentic = {k: idx2sentic[k] for k in list(idx2sentic.keys())[:min_samples]}

        # 支持三通道词表
        text_list = [line['content'] for line in lines]
        self.custom_tokenizer.fit_on_text(text_list)

        # 生成数据列表
        data = []
        for i, line in enumerate(lines):
            origin = line['content'].strip()
            pos = line['pos'] if 'pos' in line else []
            neg = line['neg'] if 'neg' in line else []
            senti = line['senti'] if 'senti' in line else []
            nonsenti = line['nonsenti'] if 'nonsenti' in line else []
            senti = senti if len(senti) > 0 else nonsenti
            sarcasm = int(line['label'])

            # 文本预处理
            content = remove_url(origin).lower()
            if self.FLAGS.tokenizer == 'spacy':
                content = [str(token) for token in nlp(content)]
            elif self.FLAGS.tokenizer == 'nltk':
                content = word_tokenize(content)
            else:
                content = content.split()

            # 计算 literal 和 deep
            if len(pos) >= len(neg) and sarcasm == 1:
                literal, deep = 1, 0
            elif len(pos) < len(neg) and sarcasm == 1:
                literal, deep = 0, 1
            elif len(pos) >= len(neg) and sarcasm == 0:
                literal, deep = 1, 1
            elif len(pos) < len(neg) and sarcasm == 0:
                literal, deep = 0, 0
            else:
                literal, deep = 0, 0

            # 填充情感词
            senti = senti if len(senti) > 0 else content
            nonsenti = nonsenti if len(nonsenti) > 0 else content

            # 生成上下文索引
            context_indices = self.custom_tokenizer.text_to_sequence(origin.strip())

            # 获取图数据（使用i作为索引）
            
            

            if len(content) > 0:
                dependency_graph = idx2graph[i] 
                sentic_graph = idx2sentic[i] 
                oneline = {
                    'content': content,
                    'sarcasm': sarcasm,
                    'senti': senti,
                    'nonsenti': nonsenti,
                    'origin': origin,
                    'literal': literal,
                    'deep': deep,
                    'context_indices': context_indices,
                    'dependency_graph': dependency_graph,
                    'sentic_graph': sentic_graph,
                    'custom_indices': [self.custom_tokenizer.word2idx.get(w, 1) for w in content]
                }
                data.append(oneline)
                
            else:
                print(i)

        np.random.shuffle(data)
        print(f"最终加载样本数: {len(data)}")
        return data

    def gen_batched_data(self, data):
        # 1. 计算统一的最大序列长度
        max_len = min(
          max(len(item['content']) for item in data),
            self.FLAGS.max_length_sen
            )
        # print(f"统一序列长度: {max_len}")


        # 2. 定义通用填充函数
        def pad_sequence(seq, target_len, pad_val='PAD'):
            if len(seq) >= target_len:
                return seq[:target_len]
            return seq + [pad_val] * (target_len - len(seq))

        def pad_graph(graph, target_len):
            if graph.shape[0] >= target_len:
                return graph[:target_len, :target_len]
            return np.pad(graph, ((0, target_len - graph.shape[0]), (0, target_len - graph.shape[0])), mode='constant')

        # 3. 处理数据
        list_sentences, list_sentis, list_nonsentis = [], [], []
        length_sen, length_senti, length_nonsenti = [], [], []
        literal, deep, sarcasm = [], [], []
        list_origin = []
        list_context_indices = []
        list_dependency_graphs = []
        list_sentic_graphs = []
        list_custom_indices, length_custom = [], []

        for item in data:
            # 填充文本序列
            sentence = pad_sequence(item['content'], max_len)
            senti = pad_sequence(item['senti'], max_len)
            nonsenti = pad_sequence(item['nonsenti'], max_len)

            # 填充图
            dependency_graph = pad_graph(item['dependency_graph'], max_len)
            sentic_graph = pad_graph(item['sentic_graph'], max_len)

            # 填充索引
            context_indices = item['context_indices'][:max_len] if len(item['context_indices']) > max_len else \
                item['context_indices'] + [0] * (max_len - len(item['context_indices']))
            
            custom_indices = item['custom_indices'][:max_len] if len(item['custom_indices']) > max_len else \
                item['custom_indices'] + [0] * (max_len - len(item['custom_indices']))

            # 收集数据
            list_sentences.append(sentence)
            list_sentis.append(senti)
            list_nonsentis.append(nonsenti)
            length_sen.append(min(max_len, len(item['content'])))
            length_senti.append(min(max_len, len(item['senti'])))
            length_nonsenti.append(min(max_len, len(item['nonsenti'])))
            literal.append(item['literal'])
            deep.append(item['deep'])
            sarcasm.append(item['sarcasm'])
            list_origin.append(item['origin'])
            list_context_indices.append(context_indices)
            list_dependency_graphs.append(dependency_graph)
            list_sentic_graphs.append(sentic_graph)
            list_custom_indices.append(custom_indices)
            length_custom.append(min(max_len, len(item['custom_indices'])))

        # 4. 构建返回数据
        batched_data = {
            'sentences': np.array(list_sentences),
            'sentis': np.array(list_sentis),
            'nonsentis': np.array(list_nonsentis),
            'origins': list_origin,
            'length_sen': np.array(length_sen),
            'length_senti': np.array(length_senti),
            'length_nonsenti': np.array(length_nonsenti),
            'literals': np.array(literal),
            'deeps': np.array(deep),
            'sarcasms': np.array(sarcasm),
            'max_len_sen': max_len,
            'context_indices': np.array(list_context_indices),
            'dependency_graphs': np.array(list_dependency_graphs),
            'sentic_graphs': np.array(list_sentic_graphs),
            'custom_indices': np.array(list_custom_indices),
            'length_custom': np.array(length_custom),
            'max_len_custom': max_len
        }
        return batched_data

    def build_vocab(self, path, data, vocab=dict()):
        logging.info('构建词汇表...')
        for inst in tqdm(data):
            for word in inst['content']:
                vocab[word] = vocab.get(word, 0) + 1

        vocab['<unk>'] = 1e9  # 确保unk在首位
        vocab_list = sorted(vocab, key=vocab.get, reverse=True)[:min(len(vocab), self.FLAGS.voc_size)]

        # 保存词汇表
        vocab_path = os.path.join(self.data_dir, 'vocab.txt')
        with open(vocab_path, 'w', encoding='utf8') as f:
            for word in vocab_list:
                f.write(word + '\n')

        # 构建嵌入矩阵
        word2idx = {word: i for i, word in enumerate(vocab_list)}
        embed = build_embedding_matrix(
            word2idx=word2idx,
            embed_dim=self.FLAGS.dim_input,
            dataset_type=self.dataset,
            data_dir=self.data_dir
        )

        vocab_wordvec = {word: None for word in vocab_list}
        return vocab_list, embed, vocab_wordvec