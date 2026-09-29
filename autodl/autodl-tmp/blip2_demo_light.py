import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ========== 方案 A: 使用 OPT-2.7B (推荐) ==========
print("\n加载 BLIP-2 OPT-2.7B 模型...")
model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_opt",
    model_type="pretrain_opt2.7b",  # 或 "caption_coco_opt2.7b"
    is_eval=True,
    device=device
)

# ========== 方案 B: 使用 Flan-T5-XL (备选) ==========
# print("\n加载 BLIP-2 Flan-T5-XL 模型...")
# model, vis_processors, txt_processors = load_model_and_preprocess(
#     name="blip2_t5",
#     model_type="pretrain_flant5xl",
#     is_eval=True,
#     device=device
# )

print("✅ 模型加载成功！")

# 测试图像路径（使用 COCO 验证集的图像）
test_image_path = "/root/autodl-tmp/datasets/coco/val2014/COCO_val2014_000000000042.jpg"

# 如果 COCO 图像不存在，创建测试图像
import os
if not os.path.exists(test_image_path):
    print("COCO 图像不存在，使用示例 URL")
    # 你也可以用 wget 下载一张测试图像
    test_image_path = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png"

# 加载图像
print(f"\n加载图像: {test_image_path}")
raw_image = Image.open(test_image_path).convert("RGB")
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# ========== 测试 1: 图像描述生成 ==========
print("\n" + "="*60)
print("测试 1: 图像描述生成")
print("="*60)

caption = model.generate({"image": image})
print(f"生成的描述: {caption[0]}")

# ========== 测试 2: 视觉问答 (VQA) ==========
print("\n" + "="*60)
print("测试 2: 视觉问答")
print("="*60)

questions = [
    "What is in the image?",
    "What color is the main object?",
    "Is this indoors or outdoors?"
]

for question in questions:
    prompt = f"Question: {question} Answer:"
    answer = model.generate({"image": image, "prompt": prompt})
    print(f"Q: {question}")
    print(f"A: {answer[0]}\n")

# ========== 测试 3: 显存使用情况 ==========
print("="*60)
print("GPU 显存使用情况:")
print("="*60)
if torch.cuda.is_available():
    print(f"已分配显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"最大显存: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"总显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

print("\n✅ 所有测试完成!")
