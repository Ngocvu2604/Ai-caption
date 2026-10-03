"""
AI Caption Server — FastAPI
Chạy model Qwen2-0.5B-Instruct một lần, phục vụ nhiều request từ Flutter App.
"""

import torch
from typing import Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn
import logging
import time

# ──────────────────────────────────────────────
# Cấu hình logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Trạng thái toàn cục — model load 1 lần duy nhất
# ──────────────────────────────────────────────
class AIEngine:
    tokenizer: Any = None
    model: Any = None
    is_ready: bool = False

engine = AIEngine()

# ──────────────────────────────────────────────
# Lifespan: load model khi server khởi động
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("⏳ Đang nạp mô hình Qwen2-0.5B-Instruct...")
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    try:
        engine.tokenizer = AutoTokenizer.from_pretrained(model_name)
        engine.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        engine.model.to("cpu")
        engine.model.eval()  # tắt dropout, tăng tốc inference
        engine.is_ready = True
        log.info("✅ AI Engine sẵn sàng phục vụ!")
    except Exception as e:
        log.error(f"❌ Không thể load model: {e}")
        engine.is_ready = False
    yield
    # Cleanup khi server tắt
    log.info("🛑 Server đang tắt, giải phóng model...")
    del engine.model
    del engine.tokenizer

# ──────────────────────────────────────────────
# Khởi tạo FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(
    title="AI Caption Gen Z API",
    description="Sinh caption Gen Z vui nhộn cho App quản lý chi tiêu 💸",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — cho phép Flutter/Android gọi từ mọi nguồn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Schema request / response
# ──────────────────────────────────────────────
class CaptionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Câu mô tả khoản chi tiêu hoặc câu chuyện của bạn",
        examples=["Hôm nay mua trà sữa 65k, đau cái ví"]
    )

class CaptionResponse(BaseModel):
    caption: str
    processing_time_ms: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str

# ──────────────────────────────────────────────
# Core: sinh caption
# ──────────────────────────────────────────────
def _generate_caption(user_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Nhiệm vụ: Viết DUY NHẤT 1 câu caption Gen Z (dưới 10 chữ) "
                "dựa trên nội dung người dùng nhập. "
                "Bám sát chủ đề chi tiêu, tiền bạc. "
                "KHÔNG HỎI LẠI. KHÔNG GIẢI THÍCH. Chỉ trả về caption."
            )
        },
        # Few-shot examples — chủ đề chi tiêu
        {"role": "user",      "content": "Ly trà sữa 65k"},
        {"role": "assistant", "content": "Uống ngụm hạnh phúc, cháy cái ví 💸🧋"},

        {"role": "user",      "content": "Ăn sáng bánh mì 20k"},
        {"role": "assistant", "content": "Đầu tư bao tử, tiết kiệm có tâm 🥖💰"},

        {"role": "user",      "content": "Mua giày mới 500k"},
        {"role": "assistant", "content": "Flex đôi chân, khóc cái ví 👟😭"},

        {"role": "user",      "content": "Hôm nay tiết kiệm không mua gì"},
        {"role": "assistant", "content": "Nhịn miệng hôm nay, giàu ngày mai ✨🔒"},

        {"role": "user",      "content": "Đổ xăng tốn 150k"},
        {"role": "assistant", "content": "Xăng leo thang, ví tụt thang 🛵💀"},

        {"role": "user",      "content": user_text},
    ]

    text = engine.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = engine.tokenizer([text], return_tensors="pt")

    with torch.no_grad():  # tắt gradient → nhanh hơn, tiết kiệm RAM
        generated_ids = engine.model.generate(
            **model_inputs,
            max_new_tokens=20,
            temperature=0.35,
            do_sample=True,
            top_p=0.85,
            repetition_penalty=1.2,
            eos_token_id=engine.tokenizer.eos_token_id,
        )

    new_ids = [
        output[len(input_ids):]
        for input_ids, output in zip(model_inputs.input_ids, generated_ids)
    ]

    caption = engine.tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    # Cắt gọn nếu quá dài
    words = caption.split()
    if len(words) > 12:
        caption = " ".join(words[:12]) + " 💀"

    return caption if caption else "Không thể sinh caption, thử lại nhé! 🔄"

# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "AI Caption API đang chạy! Xem docs tại /docs"}

@app.get("/health", response_model=HealthResponse, summary="Kiểm tra trạng thái server")
async def health_check():
    """Endpoint để Flutter/Android ping kiểm tra server còn sống không."""
    return HealthResponse(
        status="ok" if engine.is_ready else "loading",
        model_loaded=engine.is_ready,
        model_name="Qwen/Qwen2-0.5B-Instruct"
    )

@app.post("/caption", response_model=CaptionResponse, summary="Sinh caption Gen Z")
async def generate_caption(request: CaptionRequest):
    """
    Nhận câu mô tả, trả về caption Gen Z vui nhộn.

    - **text**: Câu mô tả khoản chi tiêu hoặc câu chuyện (tối đa 300 ký tự)
    """
    if not engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model đang được nạp, vui lòng thử lại sau vài giây."
        )

    start = time.time()
    try:
        caption = _generate_caption(request.text)
    except Exception as e:
        log.error(f"Lỗi sinh caption: {e}")
        raise HTTPException(
            status_code=500,
            detail="Lỗi xử lý AI. Vui lòng thử lại."
        )

    elapsed_ms = int((time.time() - start) * 1000)
    log.info(f'✅ "{request.text}" → "{caption}" ({elapsed_ms}ms)')

    return CaptionResponse(caption=caption, processing_time_ms=elapsed_ms)

# ──────────────────────────────────────────────
# Chạy server
# ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",   # lắng nghe mọi interface (kể cả từ điện thoại trong cùng mạng)
        port=8000,
        reload=False,      # tắt reload để model không load lại
        workers=1          # 1 worker vì model dùng RAM nhiều
    )
