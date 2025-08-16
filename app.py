# app.py
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
import os

# --- 全局加载模型和分词器 ---
# 我们只在程序启动时加载一次，避免每次点击都重新加载
print("Web UI启动，正在加载模型，请稍候...")

# 模型和适配器路径
base_model_path = "Qwen/Qwen1.5-7B-Chat"
lora_adapter_path = os.path.join("outputs", "final_model")

# 使用4-bit量化配置加载
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# 加载分词器和基础模型
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=quant_config,
    device_map="auto"
)

# 合并LoRA适配器
model = PeftModel.from_pretrained(base_model, lora_adapter_path)

print("模型加载完成！Gradio界面即将启动。")

# --- Gradio核心功能函数 ---
def generate_response(message, history, temperature, max_new_tokens):
    """
    Gradio的ChatInterface需要这个格式的函数。
    message: 当前用户输入。
    history: 对话历史。
    """
    # 将Gradio的history格式转换为模型需要的格式
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    # 应用聊天模板
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # 创建pipeline进行生成
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    with torch.no_grad():
        outputs = pipe(
            prompt,
            max_new_tokens=int(max_new_tokens), # 从Gradio组件获取参数
            do_sample=True,
            temperature=temperature,          # 从Gradio组件获取参数
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1
        )
    
    generated_text = outputs[0]['generated_text']
    response = generated_text[len(prompt):].strip()
    
    # 流式返回，实现打字机效果
    for char in response:
        yield char

# --- 构建Gradio界面 ---
# 使用 gr.ChatInterface，这是一个更现代的聊天界面
demo = gr.ChatInterface(
    fn=generate_response,
    title="BBS风格生成器 (LoRA微调)",
    description="一个由Qwen1.5-7B微调而来的、会模仿北大BBS风格的AI。在下方输入你的问题开始对话吧！",
    # 添加一些可调参数的组件
    additional_inputs=[
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="温度 (Temperature)"),
        gr.Slider(minimum=64, maximum=1024, value=256, step=64, label="最大新词元数 (Max New Tokens)")
    ],
    examples=[
        ["锐评一下校内网速怎么样？"],
        ["模仿北大BBS的风格，写一段关于选课的吐槽。"],
        ["我马上要毕业了，是该去大厂996还是考公上岸？"]
    ]
)

# --- 启动Web服务器 ---
if __name__ == "__main__":
    # launch() 会创建一个本地Web服务器，并提供一个URL
    demo.launch()