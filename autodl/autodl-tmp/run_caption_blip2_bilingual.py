import os, json, torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from transformers import MarianMTModel, MarianTokenizer, pipeline

# ==========================
# 1️⃣ 设备设置
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 当前设备: {device}")

# ==========================
# 2️⃣ 加载 BLIP-2 模型
# ==========================
print("🚀 正在加载 BLIP-2 模型 (pretrain_flant5xl)...")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
)
print("✅ 模型加载成功！")

# ==========================
# 3️⃣ 加载改进版翻译模型
# ==========================
print("🌏 正在加载翻译与润色模块...")
zh_model_name = "Helsinki-NLP/opus-mt-en-zh"
zh_tokenizer = MarianTokenizer.from_pretrained(zh_model_name)
zh_model = MarianMTModel.from_pretrained(zh_model_name).to(device)

# 可选：使用 pipeline 做轻量化润色
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
refine_tokenizer = AutoTokenizer.from_pretrained("uer/t5-base-chinese-cluecorpussmall")
refine_model = AutoModelForSeq2SeqLM.from_pretrained("uer/t5-base-chinese-cluecorpussmall").to(device)
print("✅ 翻译与润色模型加载成功！")

# ==========================
# 4️⃣ 图像文件夹与输出路径
# ==========================
img_dir = "/root/autodl-tmp/datasets/coco/val2014"
output_path = "results/blip2_captions_bilingual_refined.json"
os.makedirs("results", exist_ok=True)
MAX_IMAGES = 10 
# ==========================
# 5️⃣ 生成中英文描述（带润色）
# ==========================
def refine_chinese(text):
    """简单的中文润色函数"""
    prompt = f"将以下英文翻译成自然流畅的中文描述：{text}"
    inputs = refine_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = refine_model.generate(**inputs, max_length=128)
    refined = refine_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return refined.strip("。") + "。"

results = []
for fn in sorted(os.listdir(img_dir)):
    if not fn.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(img_dir, fn)
    print(f"\n🖼️ 正在处理图像: {img_path}")
    raw_image = Image.open(img_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    # 英文描述
    caption_en = model.generate({"image": image})[0]

    # 中文翻译 + 润色
    inputs = zh_tokenizer(caption_en, return_tensors="pt", padding=True).to(device)
    translated = zh_model.generate(**inputs, max_length=128)
    caption_zh = zh_tokenizer.decode(translated[0], skip_special_tokens=True)
    caption_zh = refine_chinese(caption_en) if caption_zh else refine_chinese(caption_en)

    print(f"🇬🇧 English: {caption_en}")
    print(f"🇨🇳 中文（润色后）: {caption_zh}")

    results.append({
        "image": img_path,
        "caption_en": caption_en,
        "caption_zh": caption_zh
    })

# ==========================
# 6️⃣ 保存结果
# ==========================
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
print(f"\n✅ 已保存改进版结果到: {output_path}")


