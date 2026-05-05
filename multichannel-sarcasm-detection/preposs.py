import spacy
import json

# 加载 spaCy 英文模型
nlp = spacy.load("en_core_web_sm")


# 加载 SentiWordNet
def load_sentiwordnet(path='SentiWordNet_3.0.0.txt'):
    from collections import defaultdict
    swn = defaultdict(lambda: {"pos": 0.0, "neg": 0.0})
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) != 6:
                continue
            pos_tag, _, pos_score, neg_score, synset_terms, _ = parts
            for term in synset_terms.split():
                word = term.split('#')[0].lower()
                swn[(word, pos_tag)] = {
                    "pos": float(pos_score),
                    "neg": float(neg_score)
                }
    return swn


# 映射 POS 标签
def map_pos(spacy_pos):
    if spacy_pos == "NOUN":
        return "n"
    elif spacy_pos == "VERB":
        return "v"
    elif spacy_pos == "ADJ":
        return "a"
    elif spacy_pos == "ADV":
        return "r"
    else:
        return None


# 分析单条文本
def analyze_row(text, label, swn):
    doc = nlp(text)
    pos_words = []
    neg_words = []

    for token in doc:
        swn_pos = map_pos(token.pos_)
        if swn_pos:
            senti = swn.get((token.lemma_.lower(), swn_pos), {"pos": 0.0, "neg": 0.0})
            if senti["pos"] > senti["neg"]:
                pos_words.append(token.text)
            elif senti["neg"] > senti["pos"]:
                neg_words.append(token.text)

    if len(pos_words) > len(neg_words):
        senti_label = "positive"
    elif len(neg_words) > len(pos_words):
        senti_label = "negative"
    else:
        senti_label = "neutral"


    if label == 1:
        nonsenti_label = (
            "negative" if senti_label == "positive"
            else "positive" if senti_label == "negative"
            else "neutral"
        )
    else:
        nonsenti_label = senti_label

    return pos_words, neg_words, senti_label, nonsenti_label


# 主处理函数
def process_txt(input_txt, sentiwordnet_path, output_txt):
    swn = load_sentiwordnet(sentiwordnet_path)

    with open(input_txt, 'r', encoding='utf-8') as f_in, \
            open(output_txt, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            id_, label, content = parts
            try:
                label = int(label)
                id_ = int(id_)
            except ValueError:
                continue

            pos, neg, senti, nonsenti = analyze_row(content, label, swn)
            record = {
                "id": id_,
                "content": content,
                "pos": pos,
                "neg": neg,
                "senti": [senti],
                "nonsenti": [nonsenti],
                'label':label
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"✅ 完成：处理结果已保存至 {output_txt}")


# 执行
if __name__ == '__main__':
    process_txt(
        input_txt=r'C:\Users\Lenovo\Desktop\HP\dual-channel-for-sarcasm-main\data\IAC-V2\test.csv',  # 输入你的 TXT 文件
        sentiwordnet_path=r'C:\Users\Lenovo\Desktop\HP\dual-channel-for-sarcasm-main\SentiWordNet_3.0.0.txt',
        output_txt='test.txt'  # 输出的每行 JSON 文件
    )
