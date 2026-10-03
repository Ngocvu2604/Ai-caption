"""
Test LoRA model — sinh caption Gen Z từ model đã fine-tune
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
LORA_PATH  = "./lora_caption_model"

print("⏳ Đang load model + LoRA adapter...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()
print("✅ Model sẵn sàng!\n")

def generate_caption(user_text: str) -> str:
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
        {"role": "user", "content": user_text},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.35,
            do_sample=True,
            top_p=0.85,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_ids = output[0][len(inputs.input_ids[0]):]
    caption = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return caption

# ──────────────────────────────────────────────
# Test với nhiều câu input
# ──────────────────────────────────────────────
test_inputs = [
    "Mua trà sữa 65k",
    "Đi grab hết 30k",
    "Mua áo mới 250k",
    "Ăn phở sáng 45k",
    "Nạp game 100k",
    "Tiền điện tháng này 500k",
    "Mua sách lập trình 200k",
]

print("=" * 60)
print("  🧪 TEST MODEL LoRA — Caption Gen Z")
print("=" * 60)

for text in test_inputs:
    caption = generate_caption(text)
    print(f"\n📝 Input:   {text}")
    print(f"💬 Caption: {caption}")

print("\n" + "=" * 60)
print("  ✅ Test hoàn tất!")
print("=" * 60)
