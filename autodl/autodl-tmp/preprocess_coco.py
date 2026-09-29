# preprocess_coco.py
import json
from pathlib import Path

def prepare_coco_data(ann_file, img_dir, output_file):
    """将 COCO 标注转换为训练格式"""
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # 创建图像ID到文件名的映射
    id_to_filename = {img['id']: img['file_name'] 
                      for img in data['images']}
    
    # 整理标注
    samples = []
    for ann in data['annotations']:
        img_id = ann['image_id']
        samples.append({
            'image': str(Path(img_dir) / id_to_filename[img_id]),
            'caption': ann['caption']
        })
    
    # 保存
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"Processed {len(samples)} samples to {output_file}")

# 运行
prepare_coco_data(
    'data/coco/annotations/captions_train2014.json',
    'data/coco/train2014',
    'data/coco/train_captions.json'
)

prepare_coco_data(
    'data/coco/annotations/captions_val2014.json',
    'data/coco/val2014',
    'data/coco/val_captions.json'
)

