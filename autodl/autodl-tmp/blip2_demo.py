import sys
sys.path.append("/root/autodl-tmp/LAVIS")

import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# 1️⃣ 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}, GPU 可用: {torch.cuda.is_available()}")

# 2️⃣ 加载示例图像
image_path = "blip2_illustration.png"  # 官方自带示例图
raw_image = Image.open(image_path).convert("RGB")

# 3️⃣ 加载 BLIP-2 模型 (FlanT5-XXL)
print("🔄 正在加载 BLIP-2 模型 (blip2_t5)...")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
)
print("✅ 模型加载完成！")

# 4️⃣ 预处理图像
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# 5️⃣ 模块1：图像字幕生成
print("\n🖼️ 模块1: 图像字幕生成")
caption = model.generate({"image": image})
print("👉 Caption:", caption)

# 6️⃣ 模块2：图像问答
print("\n❓ 模块2: 图像问答 (Visual Question Answering)")
question = "Question: What animal is this? Answer:"
answer = model.generate({"image": image, "prompt": question})
print("👉 Q:", question)
print("👉 A:", answer)

# 7️⃣ 模块3：指令生成
print("\n💬 模块3: 指令生成 (Prompted Generation)")
prompt = "Prompt: Describe this scene as if you were writing a travel blog."
generation = model.generate({"image": image, "prompt": prompt})
print("👉 Prompt:", prompt)
print("👉 Generated text:", generation)

print("\n🎉 所有模块执行完毕！")
