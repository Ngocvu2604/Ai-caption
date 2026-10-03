"""
train_lora.py — Fine-tune Qwen2-0.5B-Instruct với LoRA
Dùng dataset captions_dataset.json để train model chuyên về caption Gen Z chi tiêu.

Yêu cầu: pip install peft trl datasets
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import os

# ──────────────────────────────────────────────
# Cấu hình
# ──────────────────────────────────────────────
MODEL_NAME      = "Qwen/Qwen2-0.5B-Instruct"
DATASET_PATH    = "captions_dataset.json"
OUTPUT_DIR      = "./lora_caption_model"
EPOCHS          = 3
BATCH_SIZE      = 2     # nhỏ vì chạy CPU, tránh OOM
MAX_SEQ_LEN     = 256   # giới hạn độ dài sequence để tiết kiệm RAM

print("=" * 60)
print("  LoRA Fine-tune — AI Caption Gen Z Chi Tiêu")
print("=" * 60)

# ──────────────────────────────────────────────
# 1. Load & Format Dataset
# ──────────────────────────────────────────────
print("\n📂 Đang load dataset...")
with open(DATASET_PATH, encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"✅ Loaded {len(raw_data)} cặp dữ liệu")

def format_chat(sample):
    """
    Chuyển mỗi cặp {input, caption} thành định dạng chat Qwen:
    <|im_start|>system ... <|im_start|>user ... <|im_start|>assistant ...
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Nhiệm vụ: Viết DUY NHẤT 1 câu caption Gen Z (dưới 10 chữ) "
                "dựa trên nội dung người dùng nhập về chi tiêu hàng ngày. "
                "Bám sát chủ đề. Thêm emoji phù hợp. "
                "KHÔNG HỎI LẠI. KHÔNG GIẢI THÍCH. Chỉ trả về caption."
            ),
        },
        {"role": "user",      "content": sample["input"]},
        {"role": "assistant", "content": sample["caption"]},
    ]
    return messages

# Áp dụng chat template lên toàn bộ dataset
print("🔧 Đang format dataset theo chat template...")
tokenizer_temp = AutoTokenizer.from_pretrained(MODEL_NAME)

formatted_texts = []
for item in raw_data:
    msgs = format_chat(item)
    text = tokenizer_temp.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=False
    )
    formatted_texts.append({"text": text})

dataset = Dataset.from_list(formatted_texts)
print(f"✅ Dataset sẵn sàng: {len(dataset)} samples")
print(f"\n📝 Ví dụ mẫu:\n{formatted_texts[0]['text'][:300]}...\n")
del tokenizer_temp  # giải phóng RAM

# ──────────────────────────────────────────────
# 2. Load Model + Tokenizer
# ──────────────────────────────────────────────
print("⏳ Đang load model Qwen2-0.5B-Instruct...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # bắt buộc với Qwen

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,         # float32 cho CPU (float16 cần GPU)
)
print("✅ Model loaded!")

# ──────────────────────────────────────────────
# 3. Cấu hình LoRA (sẽ được SFTTrainer tự apply)
# ──────────────────────────────────────────────
print("\n🔧 Cấu hình LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)
print("✅ LoRA config sẵn sàng (sẽ được apply khi training bắt đầu)")

# ──────────────────────────────────────────────
# 4. Cấu hình Training
# ──────────────────────────────────────────────
from trl import SFTConfig

# ──────────────────────────────────────────────
# 4. Cấu hình Training (TRL 1.12 dùng SFTConfig)
# ──────────────────────────────────────────────
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=24,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=False,
    bf16=False,
    dataloader_num_workers=0,
    report_to="none",
    optim="adamw_torch",
    max_length=MAX_SEQ_LEN,          # TRL 1.12 dùng max_length (không phải max_seq_length)
    dataset_text_field="text",       # dataset_text_field thuộc SFTConfig
)

# ──────────────────────────────────────────────
# 5. Bắt đầu Train
# ──────────────────────────────────────────────
print("\n🚀 Bắt đầu fine-tune với LoRA...\n")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
    peft_config=lora_config,         # TRL 1.12: truyền peft_config vào đây
)

trainer.train()

# ──────────────────────────────────────────────
# 6. Lưu Model LoRA đã train
# ──────────────────────────────────────────────
print(f"\n💾 Đang lưu LoRA adapter vào '{OUTPUT_DIR}'...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n" + "=" * 60)
print("  ✅ FINE-TUNE HOÀN TẤT!")
print(f"  📁 Model đã lưu tại: {OUTPUT_DIR}/")
print("  👉 Bước tiếp: Cập nhật server.py để dùng model mới")
print("=" * 60)
