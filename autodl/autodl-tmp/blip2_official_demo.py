import sys
import torch
from PIL import Image

# ✅ 加入本地 LAVIS 路径
sys.path.append("/root/autodl-tmp/LAVIS")

from lavis.models import load_model_and_preprocess

# 1️⃣ 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")

# 2️⃣ 加载一张图片
image_path = "test_images/COCO_val2014_000000000042.jpg"  # 你自己的COCO图片路径
raw_image = Image.open(image_path).convert("RGB")

# 3️⃣ 加载 BLIP-2 模型（FlanT5-XXL）
print("🔄 正在加载 BLIP-2 模型 (blip2_t5, pretrain_flant5xl)...")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device
)
print("✅ 模型加载完成！")

# 4️⃣ 预处理图像
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# 5️⃣ 给出图像和问题，让模型回答（VQA）
prompt = "Question: What animal is this? Answer:"
print(f"\n❓ Prompt: {prompt}")
answer = model.generate({"image": image, "prompt": prompt})
print(f"👉 模型回答: {answer}")

# 6️⃣ 生成描述（Caption）
caption = model.generate({"image": image})
print(f"\n🖼️ 图像描述: {caption}")
