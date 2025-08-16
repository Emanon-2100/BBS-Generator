import jieba
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import re # 引入 re 模块

# --- 配置区 ---
INPUT_FILE = "bbs_corpus.txt" # 我们的原始语料库
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"  # 系统自带的微软雅黑字体
#FONT_PATH = r"C:\Users\ASUS\Desktop\BBS_Generator\simsunb.ttf"        
STOPWORDS_EXTRA = [ # 添加一些我们不希望出现在词云里的高频无意义词
    '一个', '就是', '这个', '什么', '我们', '自己', '没有', '现在', '感觉', '觉得', 
    '怎么', '但是', '还是', '这样', '那样', '可以', '因为', '所以', '如果', '还有',
    '之后', '然后', '事情', '问题', '比较', '可能', '其实', '已经', '为了', '需要'
]
OUTPUT_DIR = "data_analysis_results" # 结果输出目录

# --- 核心功能函数 ---

def load_and_clean_text(filepath):
    """加载并简单清洗文本"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"输入文件 '{filepath}' 不存在！")
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    # 移除非中文字符，只保留文本用于分析
    text = ' '.join(re.findall(r'[\u4e00-\u9fa5]+', text))
    return text

def generate_word_cloud(text, font_path, output_dir):
    """生成词云图并保存"""
    print("正在进行中文分词...")
    # 使用精确模式进行分词
    word_list = jieba.lcut(text)
    text_cut = " ".join(word_list)

    print("正在生成词云图...")
    wordcloud = WordCloud(
        width=1000,
        height=700,
        background_color='white',
        font_path=font_path,
        stopwords=set(STOPWORDS_EXTRA),
        max_words=100, # 最多显示100
        collocations=False # 避免词语重复
    ).generate(text_cut)

    plt.figure(figsize=(12, 9))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    output_path = os.path.join(output_dir, "bbs_wordcloud.png")
    plt.savefig(output_path, dpi=300) # 保存高清图
    plt.close()
    print(f"词云图已保存至: {output_path}")

def analyze_post_length(input_file, output_dir, font_path):
    """分析帖子长度分布并生成直方图"""
    with open(input_file, 'r', encoding='utf-8') as f:
        corpus = f.read()

    # 使用更智能的方式分割帖子
    # 我们可以假设两个或更多连续的换行符是一个分割点
    posts = re.split(r'\n\s*\n+', corpus)
    post_lengths = [len(post.strip()) for post in posts if len(post.strip()) > 20] # 过滤掉过短的片段
    
    if not post_lengths:
        print("未能成功分割帖子或所有帖子都太短，无法进行长度分析。")
        return

    print(f"成功分割出 {len(post_lengths)} 篇有效帖子用于长度分析。")
    print("正在分析帖子长度分布...")
    df = pd.DataFrame(post_lengths, columns=['length'])
    
    plt.figure(figsize=(10, 6))
    # 使用你指定的字体来显示中文
    plt.rcParams['font.sans-serif'] = [os.path.splitext(font_path)[0]]
    plt.rcParams['axes.unicode_minus'] = False
    
    df['length'].plot(kind='hist', bins=15, title='BBS帖子长度分布直方图', edgecolor='black')
    plt.xlabel("帖子长度 (字符数)")
    plt.ylabel("帖子数量 (频数)")
    
    output_path = os.path.join(output_dir, "post_length_distribution.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"帖子长度分布直方图已保存至: {output_path}")
    
    print("\n帖子长度统计描述:")
    print(df['length'].describe())


# --- 主程序入口 ---
if __name__ == "__main__":
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 检查字体文件是否存在
    if not os.path.exists(FONT_PATH):
        print(f"错误：找不到字体文件 '{FONT_PATH}'！请确保已将其复制到项目文件夹，并修改脚本中的FONT_PATH变量。")
    else:
        try:
            text_content = load_and_clean_text(INPUT_FILE)
            generate_word_cloud(text_content, FONT_PATH, OUTPUT_DIR)
            analyze_post_length(INPUT_FILE, OUTPUT_DIR, FONT_PATH)
            print(f"\n数据分析完成！所有结果已保存在 '{OUTPUT_DIR}' 文件夹中。")
        except Exception as e:
            print(f"处理过程中发生错误: {e}")