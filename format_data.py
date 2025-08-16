# format_data.py
import json
import re
import os

def find_separator(corpus_text):
    """智能地寻找帖子之间的分隔符"""
    # 常见的分隔符模式，可以根据你的文件内容调整
    # 我们优先寻找有多个换行符隔开的模式
    separators = ["\n\n\n", "\n---\n", "---", "仲裁团经讨论投票", "【个人情况】"]
    
    for sep in separators:
        if sep in corpus_text:
            print(f"检测到分隔符: '{repr(sep)}'")
            return sep
    
    print("警告：未找到明显的分隔符，将尝试使用两个换行符'\\n\\n'作为分隔符。")
    return "\n\n"

def format_bbs_data(input_file="bbs_corpus.txt", output_file="train_data.jsonl"):
    """将BBS语料格式化为指令微调的JSONL格式"""
    
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 '{input_file}' 不存在。请确保你已经创建了它。")
        return 

    with open(input_file, "r", encoding="utf-8") as f:
        corpus = f.read()

    # 智能寻找分隔符
    separator = find_separator(corpus)
    posts = corpus.split(separator)
    
    formatted_data = []
    post_count = 0
    for i, post in enumerate(posts):
        # 对于某些特殊分隔符，需要把分隔符本身加回到帖子的开头
        if separator in ["仲裁团经讨论投票", "【个人情况】"] and i > 0:
            post = separator + post

        post = post.strip()
        if not post or len(post) < 50: # 忽略太短的无效片段
            continue
        
        post_count += 1
        # 我们为每篇帖子创造几种不同的指令，增加数据多样性
        
        # 指令1: 总结或复述
        instruction1 = "你是一个北大BBS用户，请用BBS的风格总结或复述以下内容。"
        formatted_data.append({"instruction": instruction1, "input": post, "output": post})
        
        # 指令2: 续写或评论
        # 取帖子内容的前一部分作为输入
        prompt_content = post.split('。')[0] + '。' if '。' in post else post[:80]
        instruction2 = "你是一个北大BBS用户，请模仿下面的风格进行评论或续写。"
        formatted_data.append({"instruction": instruction2, "input": prompt_content, "output": post})

        # 指令3: 纯粹的风格模仿
        instruction3 = "请模仿北大BBS的风格，写一段帖子。"
        formatted_data.append({"instruction": instruction3, "input": "", "output": post})


    # 保存为JSONL文件
    with open(output_file, "w", encoding="utf-8") as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n数据格式化完成！")
    print(f"处理了 {post_count} 篇有效帖子，共生成 {len(formatted_data)} 条指令。")
    print(f"已保存到 {output_file}")
    
    # 打印一个样本看看
    if formatted_data:
        print("\n------------------")
        print("样本数据预览:")
        print(json.dumps(formatted_data[0], indent=2, ensure_ascii=False))
        print("------------------")

# --- 主程序入口 ---
if __name__ == "__main__":
    format_bbs_data()