# data/bbs/clean_bbs.py
import re
import os


script_dir = os.path.dirname(__file__)

# 定义输入和输出文件名
# 使用os.path.join来构建跨平台兼容的路径
input_file_path = os.path.join(script_dir, 'bbs_corpus_raw.txt')
output_file_path = os.path.join(script_dir, 'input.txt') # 我们最终的文件命名为input.txt，与莎士比亚数据集保持一致

print(f"正在从 {input_file_path} 读取原始BBS语料...")

try:
    with open(input_file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
except FileNotFoundError:
    print(f"错误：找不到原始语料文件 {input_file_path}")
    print("请确保你已经将真实的BBS语料保存为了 'bbs_corpus_raw.txt' 文件，并与此脚本放在同一个文件夹下。")
    exit()

print("开始清洗数据...")

# 1. 移除BBS元数据行，如 "发信人:", "信区:", "标  题:", "发信站:"
#    使用正则表达式，匹配以这些词开头，直到行尾的所有内容
cleaned_text = re.sub(r'^(发信人:|信区:|标\s*题:|发信站:).*?$', '', raw_text, flags=re.MULTILINE)

# 2. 移除BBS签名档的分隔符 "--"
cleaned_text = re.sub(r'^--\s*$', '', cleaned_text, flags=re.MULTILINE)

# 3. 移除官方公告的落款，如 "北大未名BBS 仲裁团"
cleaned_text = re.sub(r'^\s*北大未名BBS.*$', '', cleaned_text, flags=re.MULTILINE)
cleaned_text = re.sub(r'^\s*20\d{2}年\d{2}月\d{2}日\s*$', '', cleaned_text, flags=re.MULTILINE)

# 4. 移除引用部分（以">"或"【 在 ... 的大作中提到: 】"开头的行）
cleaned_text = re.sub(r'^>.*$', '', cleaned_text, flags=re.MULTILINE)
cleaned_text = re.sub(r'^【 在.*的大作中提到: 】.*$', '', cleaned_text, flags=re.MULTILINE)

# 5. 移除URL链接
cleaned_text = re.sub(r'https?://\S+', '', cleaned_text)
# 移除邮件地址
cleaned_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', cleaned_text)

# 6. 将多个连续的换行符合并为一个，以保持段落感
cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)

# 7. 去除行首和行尾的空白字符
cleaned_text = '\n'.join([line.strip() for line in cleaned_text.split('\n') if line.strip()])

print(f"数据清洗完成！正在将结果写入 {output_file_path}...")

with open(output_file_path, 'w', encoding='utf-8') as f:
    f.write(cleaned_text)

print(f"清洗脚本执行完毕！请检查生成的 '{output_file_path}' 文件。")