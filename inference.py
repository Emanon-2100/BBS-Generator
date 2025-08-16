# inference.py (最终稳定版)
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from peft import PeftModel
import os

# --- 1. 路径和配置 ---
base_model_path = "Qwen/Qwen1.5-7B-Chat"
lora_adapter_path = os.path.join("outputs", "final_model")

# --- 2. 使用和训练时完全相同的4-bit量化配置 ---
# 这能保证模型一定能被成功加载
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# --- 3. 加载分词器和量化后的模型 ---
print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # 确保pad_token设置正确

print("正在以4-bit量化方式加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=quant_config, # 使用我们的量化配置
    device_map="auto"
)

print("正在将LoRA适配器合并到基础模型中...")
# 使用PeftModel将LoRA权重合并到量化后的基础模型上
model = PeftModel.from_pretrained(base_model, lora_adapter_path)

print("\n--- 模型准备就绪！BBS风格生成器启动 ---")
print("--- 输入你的指令，输入 'exit' 或 'quit' 退出 ---")

# --- 4. 创建交互式对话 ---
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

while True:
    user_input = input("你: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    # 应用Qwen1.5的聊天模板
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    print("模型正在思考...")

    with torch.no_grad(): # 推理时不需要计算梯度
        outputs = pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1
        )
    
    generated_text = outputs[0]['generated_text']
    response = generated_text[len(prompt):].strip()

    print(f"BBS机器人: {response}\n")