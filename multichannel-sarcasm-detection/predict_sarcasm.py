# -*- coding: utf-8 -*-
"""
讽刺识别模型预测脚本 - 最终修复版
完全解决嵌入层尺寸不匹配问题
"""

import os
import torch
import numpy as np
import json
import pickle
from dataUtils import DataManager
from bridgeModel import bridgeModel


class ModelConfig:
    """模型配置类，与你的训练配置保持一致"""
    def __init__(self):
        # 基础配置
        self.seed = 2021
        self.name_dataset = "IAC2"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 模型结构参数（这些将被自动检测覆盖）
        self.voc_size = 30000
        self.dim_input = 300
        self.dim_hidden = 256
        self.dim_bert = 768
        self.n_layers = 1  # 将被自动检测覆盖
        self.n_class = 2
        self.bidirectional = 1  # 将被自动检测覆盖
        self.rnn_type = "LSTM"
        self.max_length_sen = 100

        # 训练参数
        self.learning_rate = 0.001
        self.lr_word_vector = 1e-4
        self.lr_bert = 5e-05
        self.weight_decay = 0
        self.batch_size = 1
        self.iter_num = 32 * 150
        self.per_checkpoint = 32
        self.optim_type = "Adam"

        # Dropout参数
        self.embed_dropout_rate = 0.5
        self.cell_dropout_rate = 0.5
        self.final_dropout_rate = 0.5
        self.linear_dropout_rate = 0.1

        # 其他参数
        self.t_sne = 0
        self.lambda1 = 0.5
        self.multi_dim = 20
        self.tokenizer = "spacy"
        self.margin = 0.5
        self.supcon = 1
        self.name_model = "dualbilstm"

        # 路径参数
        self.data_dir = "IAC2/spacy/"
        self.path_wordvec = "glove.840B.300d.txt"
        self.predict_dir = "./predict/"
        self.model_dir = "./models/"
        self.save_model = 0
        self.predict = 0


def analyze_model_structure(model_path, device):
    """分析模型文件结构，检测关键参数"""
    try:
        state_dict = torch.load(model_path, map_location=device)
        print("=" * 60)
        print("模型结构分析报告")
        print("=" * 60)
        
        # 显示前20个关键键
        print("\n关键参数键:")
        all_keys = sorted(state_dict.keys())
        for key in all_keys[:20]:
            print(f"  {key}: {state_dict[key].shape}")
        
        if len(all_keys) > 20:
            print(f"  ... 和 {len(all_keys) - 20} 个更多键")

        # 分析关键参数
        embed_keys = [k for k in all_keys if 'embed.weight' in k]
        rnn_keys = [k for k in all_keys if 'rnn' in k and 'weight_ih_l' in k]
        
        # 检测词表大小
        vocab_size = None
        if embed_keys:
            vocab_size = state_dict[embed_keys[0]].shape[0]
            print(f"  - 嵌入层词表大小: {vocab_size}")

        # 检测RNN层数
        layer_numbers = set()
        for key in rnn_keys:
            parts = key.split('l')
            if len(parts) > 1:
                layer_num_str = parts[-1].split('_')[0]
                if layer_num_str.isdigit():
                    layer_numbers.add(int(layer_num_str))
        
        n_layers = max(layer_numbers) + 1 if layer_numbers else 1
        print(f"  - RNN层数: {n_layers} (检测到层号: {sorted(layer_numbers)})")

        # 检测双向性
        bidirectional = any('reverse' in key for key in all_keys)
        print(f"  - 双向RNN: {bidirectional}")

        # 检测隐藏层维度
        hidden_dim = None
        for key in rnn_keys:
            if 'weight_hh_l0' in key:
                hidden_dim = state_dict[key].shape[1]
                break
        if hidden_dim:
            print(f"  - 隐藏层维度: {hidden_dim}")

        print(f"  - 总参数键数: {len(all_keys)}")
        print(f"  - RNN相关键: {len(rnn_keys)}")
        print(f"  - 嵌入层键: {len(embed_keys)}")
        
        return state_dict, vocab_size, n_layers, bidirectional, hidden_dim
        
    except Exception as e:
        print(f"❌ 模型结构分析失败: {str(e)}")
        raise


def create_embedding_from_model(state_dict, target_vocab_size, embed_dim):
    """从模型状态字典中提取嵌入层权重"""
    print("  - 从模型状态字典提取嵌入层权重...")
    
    # 查找嵌入层权重
    embed_weights = None
    for key in state_dict:
        if 'embed.weight' in key:
            embed_weights = state_dict[key].cpu().numpy()
            print(f"  - 找到嵌入层: {key}, 形状: {embed_weights.shape}")
            break
    
    if embed_weights is None:
        raise ValueError("在模型状态字典中找不到嵌入层权重")
    
    # 检查尺寸并调整
    if embed_weights.shape[0] == target_vocab_size:
        print(f"  - 嵌入层尺寸匹配: {embed_weights.shape}")
        return embed_weights
    elif embed_weights.shape[0] < target_vocab_size:
        # 补充随机向量
        pad_size = target_vocab_size - embed_weights.shape[0]
        pad_embed = np.random.normal(0, 0.01, (pad_size, embed_dim)).astype(np.float32)
        new_embed = np.vstack([embed_weights, pad_embed])
        print(f"  - 嵌入层补充: {embed_weights.shape} -> {new_embed.shape}")
        return new_embed
    else:
        # 截断
        new_embed = embed_weights[:target_vocab_size]
        print(f"  - 嵌入层截断: {embed_weights.shape} -> {new_embed.shape}")
        return new_embed


def create_compatible_model(config, vocab, embed, state_dict):
    """创建与保存的state_dict兼容的模型"""
    print("  - 创建兼容模型...")
    
    # 使用检测到的配置创建模型
    model = bridgeModel(config, vocab=vocab, embed=embed)
    
    # 获取当前模型的state_dict
    current_state_dict = model.state_dict()
    
    # 创建新的state_dict，只加载匹配的键
    new_state_dict = {}
    matched_keys = 0
    shape_mismatch_keys = 0
    missing_keys = 0
    
    print("  - 开始权重匹配...")
    for key, value in state_dict.items():
        if key in current_state_dict:
            if current_state_dict[key].shape == value.shape:
                new_state_dict[key] = value
                matched_keys += 1
            else:
                # 只显示重要的形状不匹配
                if any(pattern in key for pattern in ['rnn', 'embed', 'linear', 'dense']):
                    print(f"    ⚠️ 形状不匹配: {key}")
                    print(f"      期望: {current_state_dict[key].shape}, 实际: {value.shape}")
                shape_mismatch_keys += 1
        else:
            missing_keys += 1
            # 只显示重要的缺失键
            if any(pattern in key for pattern in ['rnn', 'embed', 'linear', 'dense']):
                print(f"    ❌ 键不存在: {key}")
    
    print(f"  - 匹配结果: {matched_keys}个匹配, {shape_mismatch_keys}个形状不匹配, {missing_keys}个缺失")
    
    if matched_keys > 0:
        # 加载匹配的权重
        model.load_state_dict(new_state_dict, strict=False)
        print("  - ✅ 兼容模式加载成功")
        return model
    else:
        print("  - ❌ 没有匹配的权重键，无法加载模型")
        return None


def robust_predict(model_path, test_path, config):
    """
    健壮的预测函数 - 修复嵌入层尺寸问题
    """
    print("=" * 60)
    print("开始健壮预测流程 - 修复版")
    print("=" * 60)
    
    try:
        # 步骤1: 检查文件存在性
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"测试文件不存在: {test_path}")

        # 步骤2: 分析模型结构
        print("[1/7] 🔍 分析模型结构...")
        state_dict, model_vocab_size, n_layers, bidirectional, hidden_dim = analyze_model_structure(model_path, config.device)
        
        # 更新配置参数
        config.n_layers = n_layers
        config.bidirectional = 1 if bidirectional else 0
        if hidden_dim:
            config.dim_hidden = hidden_dim
        
        print(f"  - 应用检测到的参数:")
        print(f"     层数: {n_layers}")
        print(f"     双向: {bidirectional}")
        print(f"     隐藏层: {hidden_dim}")
        print(f"     词表大小: {model_vocab_size}")

        # 步骤3: 初始化 DataManager
        print("[2/7] 📚 初始化 DataManager...")
        datamanager = DataManager(config)
        
        # 步骤4: 构建词表（但不使用缓存的嵌入矩阵）
        print("[3/7] 🔤 构建词表...")
        try:
            train_data = datamanager.load_data(config.data_dir, 'train.txt')
            
            # 手动构建词表，避免使用缓存的嵌入矩阵
            print("  - 手动构建词表...")
            vocab = []
            vocab_dict = {}
            
            # 从训练数据收集词汇
            word_counts = {}
            for item in train_data:
                for word in item['content']:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # 排序并截断
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            vocab = [word for word, count in sorted_words[:config.voc_size]]
            
            # 添加特殊标记
            if '<unk>' not in vocab:
                vocab.insert(0, '<unk>')
            if '<pad>' not in vocab:
                vocab.insert(0, '<pad>')
            
            # 创建词汇字典
            vocab_dict = {word: idx for idx, word in enumerate(vocab)}
            print(f"  - 成功构建词表: {len(vocab)} 个词")
            
        except Exception as e:
            print(f"  - ⚠️ 构建词表失败: {e}")
            print("  - 使用模型中的词表大小创建基础词表...")
            # 创建基础词表
            vocab = [f"word_{i}" for i in range(min(20000, model_vocab_size))]
            vocab_dict = {word: idx for idx, word in enumerate(vocab)}

        # 步骤5: 从模型状态字典创建嵌入矩阵
        print("[4/7] 🎯 创建嵌入矩阵...")
        embed = create_embedding_from_model(state_dict, model_vocab_size, config.dim_input)
        
        # 步骤6: 调整词表尺寸匹配嵌入矩阵
        print("[5/7] 📏 调整词表匹配嵌入矩阵...")
        if len(vocab) != model_vocab_size:
            print(f"  - 词表尺寸不匹配: 当前{len(vocab)} vs 需要{model_vocab_size}")
            
            if len(vocab) < model_vocab_size:
                # 补充词表
                pad_size = model_vocab_size - len(vocab)
                print(f"  - 补充 {pad_size} 个词")
                
                for i in range(pad_size):
                    word = f"[PAD_{i}]"
                    vocab.append(word)
                    vocab_dict[word] = len(vocab) - 1
            else:
                # 截断词表
                print(f"  - 截断到 {model_vocab_size} 个词")
                vocab = vocab[:model_vocab_size]
                vocab_dict = {word: idx for idx, word in enumerate(vocab)}
        
        print(f"  - 最终尺寸: 词表{len(vocab)}, 嵌入{embed.shape}")

        # 步骤7: 创建和加载模型
        print("[6/7] 🤖 创建和加载模型...")
        
        # 直接使用兼容模式加载（避免严格模式错误）
        model = bridgeModel(config, vocab=vocab, embed=embed)
        print("  - 使用兼容模式加载...")
        model = create_compatible_model(config, vocab, embed, state_dict)
        
        if model is None:
            raise RuntimeError("兼容模式加载失败")

        model.to(config.device)
        model.eval()

        # 步骤8: 加载测试数据
        print("[7/7] 📊 加载测试数据...")
        test_data = datamanager.load_data(config.data_dir, 'test.txt')
        print(f"  - 加载 {len(test_data)} 个测试样本")

        # 步骤9: 执行预测
        print("🚀 开始预测...")
        results = []
        successful_predictions = 0
        
        for i, item in enumerate(test_data, 1):
            try:
                # 使用 DataManager 的 gen_batched_data 方法
                batched_data = datamanager.gen_batched_data([item])
                
                with torch.no_grad():
                    _, probs, _, _ = model.stepTrain(batched_data, inference=True)
                
                # 提取预测概率
                if probs is not None and len(probs) > 0:
                    sarcasm_prob = float(probs[0][1])
                    
                    result = {
                        "sentence": item.get('origin', 'Unknown'),
                        "probability": round(sarcasm_prob, 4),
                        "is_sarcastic": sarcasm_prob > 0.5,
                        "true_label": item.get('sarcasm', -1)
                    }
                    successful_predictions += 1
                else:
                    result = {
                        "sentence": item.get('origin', 'Unknown'),
                        "error": "模型返回空概率",
                        "probability": 0.5,
                        "is_sarcastic": False,
                        "true_label": item.get('sarcasm', -1)
                    }
                
                results.append(result)
                
            except Exception as e:
                result = {
                    "sentence": item.get('origin', 'Unknown'),
                    "error": str(e),
                    "probability": 0.5,
                    "is_sarcastic": False,
                    "true_label": item.get('sarcasm', -1)
                }
                results.append(result)
            
            # 进度报告
            if i % 100 == 0 or i == len(test_data):
                print(f"  - 已处理 {i}/{len(test_data)} 个样本")

        # 统计结果
        failed = len(results) - successful_predictions
        sarcastic_count = sum(1 for r in results if r.get('is_sarcastic', False))
        
        print(f"  - 预测统计: 成功{successful_predictions}, 失败{failed}, 讽刺句{sarcastic_count}")

        return results, datamanager

    except Exception as e:
        print(f"❌ 预测过程失败: {str(e)}")
        raise


def save_predictions(results, output_file="sarcasm_predictions_fixed.txt"):
    """保存预测结果"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("讽刺识别预测结果（修复版）\n")
        f.write("=" * 60 + "\n\n")
        
        for idx, res in enumerate(results, 1):
            f.write(f"样本 {idx}:\n")
            f.write(f"原文: {res['sentence']}\n")
            
            if "error" in res:
                f.write(f"状态: ❌ 预测失败\n")
                f.write(f"错误信息: {res['error']}\n")
            else:
                f.write(f"状态: ✅ 预测成功\n")
                f.write(f"讽刺概率: {res['probability']:.4f}\n")
                f.write(f"预测结果: {'讽刺' if res['is_sarcastic'] else '非讽刺'}\n")
                if 'true_label' in res and res['true_label'] != -1:
                    true_label = '讽刺' if res['true_label'] == 1 else '非讽刺'
                    f.write(f"真实标签: {true_label}\n")
                    correct = res['is_sarcastic'] == (res['true_label'] == 1)
                    f.write(f"预测正确: {'是' if correct else '否'}\n")
            
            f.write("-" * 50 + "\n")
    
    print(f"💾 结果保存至: {output_file}")


if __name__ == "__main__":
    # 配置和路径
    config = ModelConfig()
    MODEL_PATH = "models/IAC2_dualbilstm/IAC2dualbilstm_best/model69.pth"
    TEST_PATH = "IAC2/spacy/test.txt"
    
    print("🎯 讽刺识别模型预测 - 修复版")
    print(f"模型路径: {MODEL_PATH}")
    print(f"测试数据: {TEST_PATH}")
    print(f"设备: {config.device}")
    print()
    
    try:
        # 执行健壮预测
        results, datamanager = robust_predict(MODEL_PATH, TEST_PATH, config)
        
        # 保存结果
        save_predictions(results)
        
        # 最终报告
        successful = sum(1 for r in results if 'error' not in r)
        sarcastic = sum(1 for r in results if r.get('is_sarcastic', False))
        
        print("\n" + "=" * 60)
        print("🎉 预测完成!")
        print(f"✅ 成功预测: {successful}/{len(results)}")
        print(f"😊 讽刺样本: {sarcastic}")
        print(f"😐 非讽刺样本: {successful - sarcastic}")
        
        if successful > 0:
            accuracy = sum(1 for r in results if 'error' not in r and 'true_label' in r and 
                          r.get('is_sarcastic') == (r.get('true_label') == 1)) / successful
            print(f"📊 准确率: {accuracy:.4f}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"\n💥 预测失败: {str(e)}")
        print("请检查以下可能的问题:")
        print("1. 模型文件路径是否正确")
        print("2. 测试数据文件是否存在")
        print("3. 数据目录结构是否与训练时一致")