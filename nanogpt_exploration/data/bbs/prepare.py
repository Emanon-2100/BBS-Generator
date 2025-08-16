# data/bbs/prepare.py
import os
import tiktoken
import numpy as np

# --- 数据源和分词器设置 ---
# 我们要处理的数据文件
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
# 我们要使用的分词器模型，基于GPT-4的，效果很好
encoding_name = 'cl100k_base' 

# --- 读取数据 ---
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
# 划分训练集和验证集 (95% 训练, 5% 验证)
train_data = data[:int(n*0.95)]
val_data = data[int(n*0.95):]

# --- 编码/分词 ---
print(f"正在使用 '{encoding_name}' 分词器进行编码...")
enc = tiktoken.get_encoding(encoding_name)
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"训练集: {len(train_ids)} tokens")
print(f"验证集: {len(val_ids)} tokens")

# 导出为 .bin 文件
train_ids = np.array(train_ids, dtype=np.uint32) # uint16 不够用
val_ids = np.array(val_ids, dtype=np.uint32)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# --- 保存元数据 ---
# 在这个方案里，我们不需要像莎士比亚数据集那样保存meta.pkl
# 因为tiktoken的分词器是标准化的，不需要我们自己维护词汇表
# train.py在加载时会自动处理这种情况
print("预处理完成！已生成 train.bin 和 val.bin 文件。")