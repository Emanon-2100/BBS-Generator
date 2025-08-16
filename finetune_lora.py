# finetune_lora.py
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# --- 1. 选择模型 ---
# 我们选择一个在中文上表现出色且对硬件友好的模型
# Qwen1.5-7B-Chat 是一个很好的选择
model_name = "Qwen/Qwen1.5-7B-Chat"

# --- 2. 数据集路径 ---
# 就是我们昨天生成的jsonl文件
dataset_name = "train_data.jsonl"

# --- 3. LoRA 和量化配置 ---
# LoRA 配置
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

# BitsAndBytes 量化配置 (在4060上运行7B模型的关键)
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# --- 4. 训练参数 ---
output_dir = "./outputs"       # 训练结果的输出目录
num_train_epochs = 1           # 训练轮次。对于高质量小数据集，1-3轮即可
fp16 = False                   # 是否使用fp16/bf16混合精度
bf16 = True                    # 在40系列显卡上，bf16性能更好
per_device_train_batch_size = 1 # 每个GPU的批处理大小
per_device_eval_batch_size = 1
gradient_accumulation_steps = 2 # 梯度累积步数，变相扩大batch size
gradient_checkpointing = True   # 使用梯度检查点，节省显存
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1                  # 如果设置为正数，将覆盖 num_train_epochs
warmup_ratio = 0.03
group_by_length = True          # 将相似长度的序列打包在一起，提高效率
save_steps = 25                 # 每25步保存一次检查点
logging_steps = 5               # 每5步打印一次日志

# --- 5. 开始训练 ---
# 加载数据集
dataset = load_dataset("json", data_files=dataset_name, split="train")

# 配置量化
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# 检查GPU兼容性
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("你的GPU支持 bfloat16，训练效率会更高。")
        print("=" * 80)

# 加载基础模型
print(f"正在从Hugging Face Hub加载模型: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto" # 自动将模型分配到可用的GPU上
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 配置PEFT (LoRA)
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[ # 针对Qwen1.5模型的特定层
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# 设置训练参数
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard" # 你可以稍后用tensorboard查看loss曲线
)

# 初始化SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="instruction", # SFTTrainer需要知道哪个字段是我们的指令
    max_seq_length=1024,              # 最大序列长度
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False, # 如果数据集小，最好关闭packing
)

# 开始训练！
print("--- 开始微调 ---")
trainer.train()
print("--- 微调完成 ---")

# 保存最终的模型
final_model_path = os.path.join(output_dir, "final_model")
trainer.model.save_pretrained(final_model_path)
print(f"最终LoRA适配器已保存至: {final_model_path}")