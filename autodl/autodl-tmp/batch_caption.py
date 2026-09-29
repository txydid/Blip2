import sys
import os
sys.path.append("/root/autodl-tmp/LAVIS") 
import os
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")

# 加载 BLIP-2 模型
print("🔄 正在加载 BLIP-2 模型 (blip2_t5)...")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
)
print("✅ 模型加载完成！")

# 图片目录
image_dir = "./test_images"
output_file = "captions_results.txt"

# 打开输出文件
with open(output_file, "w") as f:
    # 遍历所有图片
    for filename in sorted(os.listdir(image_dir)):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        image_path = os.path.join(image_dir, filename)
        raw_image = Image.open(image_path).convert("RGB")

        # 预处理
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # 生成 Caption
        caption = model.generate({"image": image})[0]

        print(f"🖼️ {filename} -> {caption}")
        f.write(f"{filename}\t{caption}\n")

print(f"\n🎉 全部生成完成！结果已保存至: {output_file}")
