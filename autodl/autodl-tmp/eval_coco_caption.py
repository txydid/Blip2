import json
import os
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

# === 路径设置 ===
result_file = "./coco_val_captions.json"
annotation_file = "/root/autodl-tmp/datasets/coco/annotations/captions_val2014.json"

print("📂 Annotation file:", annotation_file)
print("📄 Result file:", result_file)

# === 加载结果文件 ===
with open(result_file, "r") as f:
    results = json.load(f)
pred_image_ids = {item["image_id"] for item in results}
print(f"✅ Found {len(pred_image_ids)} predicted images")

# === 只保留这些ID的标注 ===
coco = COCO(annotation_file)
ann_ids = [ann_id for ann_id, ann in coco.anns.items() if ann["image_id"] in pred_image_ids]
anns_filtered = [coco.anns[i] for i in ann_ids]
coco.dataset["annotations"] = anns_filtered
coco.createIndex()

# === 加载结果并评测 ===
coco_result = coco.loadRes(results)
coco_eval = COCOEvalCap(coco, coco_result)

print("🚀 开始计算 COCO caption 指标 ...")
coco_eval.evaluate()

print("\n🎯 评测结果:")
for metric, score in coco_eval.eval.items():
    print(f"{metric}: {score:.3f}")

with open("./coco_eval_results.json", "w") as f:
    json.dump(coco_eval.eval, f, indent=2)
print("\n✅ 已保存评测结果到: ./coco_eval_results.json")

