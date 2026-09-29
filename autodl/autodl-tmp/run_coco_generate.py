import os
import json
from tqdm import tqdm
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
print("Loading BLIP-2 model...")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device
)
model.eval()
print("✅ Model loaded.")

# 数据路径
img_dir = "/root/autodl-tmp/datasets/coco/val2014"
save_path = "./coco_val_captions.json"

# 获取部分图像（比如前100张）
img_list = sorted(os.listdir(img_dir))[:5000]

results = []
for img_name in tqdm(img_list, desc="Generating captions"):
    img_path = os.path.join(img_dir, img_name)
    try:
        image = Image.open(img_path).convert("RGB")
        image = vis_processors["eval"](image).unsqueeze(0).to(device)
        samples = {"image": image, "prompt": "a photo of"}
        with torch.no_grad():
            caption = model.generate(samples, max_length=30, num_beams=3)[0]
        results.append({
            "image_id": int(img_name.split("_")[-1].split(".")[0]),
            "caption": caption
        })
    except Exception as e:
        print(f"⚠️ Error on {img_name}: {e}")

# 保存结果
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Saved captions to {save_path}")
