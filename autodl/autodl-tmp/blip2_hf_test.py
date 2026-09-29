"""BLIP-2 推理测试 - Hugging Face 版本（无需 LAVIS）"""
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import glob

print("="*60)
print("BLIP-2 推理测试 (Hugging Face 直接调用)")
print("="*60)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✓ 设备: {device}")

# 显示 GPU 信息
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ 显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

# 加载模型（OPT-2.7B - 适合 24GB 显存）
print("\n正在加载 BLIP-2 OPT-2.7B...")
print("(首次运行会下载模型，约 5GB，请耐心等待)")

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,  # 使用 FP16 节省显存
    device_map="auto"
)

print("✅ 模型加载成功!")

# 显存使用情况
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    print(f"✓ 已使用显存: {allocated:.2f} GB")

# 测试图像
coco_path = "/root/autodl-tmp/datasets/coco/val2014"
test_images = glob.glob(f"{coco_path}/*.jpg")[:5]

if not test_images:
    print("⚠️  未找到 COCO 图像，跳过测试")
    exit(0)

print(f"\n找到 {len(test_images)} 张测试图像")

# 测试 1: 图像描述生成
print("\n" + "="*60)
print("测试 1: 图像描述生成")
print("="*60)

for idx, img_path in enumerate(test_images[:3], 1):
    print(f"\n[{idx}] 图像: {img_path.split('/')[-1]}")
    
    image = Image.open(img_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=30)
    
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(f"    生成描述: {caption}")

# 测试 2: 视觉问答
print("\n" + "="*60)
print("测试 2: 视觉问答 (VQA)")
print("="*60)

test_image = Image.open(test_images[0]).convert('RGB')
questions = [
    "What is in the image?",
    "What color is the main object?",
    "Is this indoors or outdoors?"
]

for q in questions:
    prompt = f"Question: {q} Answer:"
    inputs = processor(images=test_image, text=prompt, return_tensors="pt").to(device, torch.float16)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=20)
    
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(f"\nQ: {q}")
    print(f"A: {answer}")

# 最终显存统计
print("\n" + "="*60)
print("GPU 显存统计")
print("="*60)
if torch.cuda.is_available():
    print(f"当前分配: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
    print(f"峰值使用: {torch.cuda.max_memory_allocated(0)/1024**3:.2f} GB")
    print(f"总显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

print("\n" + "="*60)
print("✅ 所有测试完成!")
print("="*60)
