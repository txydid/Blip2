# evaluate_caption_blip2.py
import os, json, torch
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from lavis.datasets.builders import load_dataset
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 当前设备: {device}")

# --- 根据你已下载并能加载的模型改这里 ---
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xl",
    is_eval=True,
    device=device
)
print("✅ 模型加载成功！")

# --- 加载 COCO 验证集（LAVIS 的 load_dataset） ---
datasets = load_dataset("coco_caption")  # 返回 dict: train/val/test
val_dataset = datasets["val"]
print(f"✅ 数据集加载成功，样本数: {len(val_dataset)}")

# ---------- 生成 Caption ----------
prompt = "Describe this image in detail."
batch_size = 2
out_file = "coco_val_preds.json"
results = []
imgs_batch, ids_batch = [], []

for idx in tqdm(range(len(val_dataset)), desc="🧠 Generating captions"):
    sample = val_dataset[idx]
    pil_img = sample["image"]
    img_tensor = vis_processors["eval"](pil_img).unsqueeze(0)
    imgs_batch.append(img_tensor)
    ids_batch.append(int(sample["image_id"]))

    if len(imgs_batch) == batch_size or idx == len(val_dataset) - 1:
        batch_tensor = torch.cat(imgs_batch, dim=0).to(device)
        with torch.no_grad():
            outputs = model.generate({"image": batch_tensor, "prompt": prompt})
        for image_id, cap in zip(ids_batch, outputs):
            results.append({"image_id": int(image_id), "caption": cap})
        imgs_batch, ids_batch = [], []

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"✅ 所有预测结果已保存到 {out_file}")

# ---------- COCO 官方评估（只对预测过的图片子集评估） ----------
annFile = "/root/autodl-tmp/datasets/coco/annotations/captions_val2014.json"
if not os.path.exists(annFile):
    raise FileNotFoundError(f"❌ 找不到标注文件: {annFile}")

print("🧾 正在执行 COCO 官方评估...")

coco_full = COCO(annFile)
with open(out_file, "r", encoding="utf-8") as f:
    preds = json.load(f)
pred_ids = [int(p["image_id"]) for p in preds]

# 只构造包含你预测图像的子集标注（避免 keys mismatch）
sub_ann_ids = coco_full.getAnnIds(imgIds=pred_ids)
sub_anns = coco_full.loadAnns(sub_ann_ids)
sub_coco = COCO()
sub_coco.dataset = {
    'images': coco_full.loadImgs(pred_ids),
    'annotations': sub_anns,
    'type': 'captions',
    'info': coco_full.dataset.get('info', {}),
    'licenses': coco_full.dataset.get('licenses', [])
}
sub_coco.createIndex()

cocoRes = sub_coco.loadRes(out_file)
cocoEval = COCOEvalCap(sub_coco, cocoRes)
cocoEval.evaluate()

print("\n===== 🧮 COCO Caption Evaluation Scores =====")
for metric, score in cocoEval.eval.items():
    print(f"{metric}: {score:.4f}")
print("✅ 评估完成！")
