"""
BLIP-2 双语图像描述生成器
支持中文和英文描述
"""
from lavis.models import load_model_and_preprocess
from PIL import Image
import torch
import os
from datetime import datetime

print("=" * 60)
print("🌍 BLIP-2 双语图像描述生成器")
print("=" * 60)

# 加载模型
print("\n📥 加载模型...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device
)
print("✅ 模型加载完成！")

def generate_caption(image_path, emoji="🖼️"):
    """
    生成图像的中英文描述
    
    Args:
        image_path: 图像文件路径
        emoji: 显示的表情符号
    """
    # 加载图像
    raw_image = Image.open(image_path).convert('RGB')
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    
    # 生成英文描述
    en_caption = model.generate({"image": image})[0]
    
    # 生成中文描述（通过提示词引导）
    zh_prompt = "用中文描述这张图片:"
    zh_caption = model.generate({
        "image": image,
        "prompt": zh_prompt
    })[0]
    
    # 如果中文生成失败，使用简单翻译提示
    if not any('\u4e00' <= char <= '\u9fff' for char in zh_caption):
        # 没有中文字符，尝试另一种方式
        zh_prompt2 = "Translate to Chinese: " + en_caption
        zh_caption = model.generate({
            "image": image,
            "prompt": zh_prompt2
        })[0]
    
    # 输出结果
    print(f"{emoji} 图像 → EN: '{en_caption}'")
    print(f"    ZH: '{zh_caption}'")
    
    return en_caption, zh_caption

def process_directory(image_dir, max_images=10):
    """
    批量处理目录中的图像
    
    Args:
        image_dir: 图像目录路径
        max_images: 最多处理的图像数量
    """
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    image_files = image_files[:max_images]
    
    # 表情符号列表
    emojis = ["☕", "🐶", "🚗", "🏠", "🌸", "🍕", "📱", "⚽", "🎨", "🌈"]
    
    results = []
    
    print(f"\n🚀 开始处理 {len(image_files)} 张图像...\n")
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        emoji = emojis[idx % len(emojis)]
        
        try:
            en_caption, zh_caption = generate_caption(img_path, emoji)
            results.append({
                'file': img_file,
                'en': en_caption,
                'zh': zh_caption
            })
        except Exception as e:
            print(f"❌ 处理 {img_file} 失败: {e}")
            continue
    
    return results

def save_results(results, output_file="captions_bilingual.txt"):
    """保存结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for idx, result in enumerate(results, 1):
            f.write(f"{idx}. 文件: {result['file']}\n")
            f.write(f"   英文: {result['en']}\n")
            f.write(f"   中文: {result['zh']}\n")
            f.write("\n")
    
    print(f"\n💾 结果已保存到: {output_file}")

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 方式 1：处理单张图像
    print("\n" + "=" * 60)
    print("📸 示例 1: 单张图像")
    print("=" * 60)
    
    sample_image = "/root/autodl-tmp/datasets/coco/val2014/COCO_val2014_000000522418.jpg"
    if os.path.exists(sample_image):
        generate_caption(sample_image, "☕")
    
    # 方式 2：批量处理目录
    print("\n" + "=" * 60)
    print("📸 示例 2: 批量处理")
    print("=" * 60)
    
    image_dir = "/root/autodl-tmp/datasets/coco/val2014"
    results = process_directory(image_dir, max_images=5)
    
    # 保存结果
    save_results(results)
    
    print("\n✅ 完成！")

