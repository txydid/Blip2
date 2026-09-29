from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from PIL import Image
import json
import torch
import os

class SmartCOCODataset(Dataset):
    def __init__(self, ann_file, img_dir, processor, max_samples=5000):
        print(f"📂 加载标注文件: {ann_file}")
        with open(ann_file, 'r') as f:
            all_data = json.load(f)
        
        print(f"📊 标注文件总数: {len(all_data)}")
        
        # 统计
        train2014_total = sum(1 for item in all_data if 'train2014' in item['image'])
        val2014_total = sum(1 for item in all_data if 'val2014' in item['image'])
        print(f"   - train2014: {train2014_total}")
        print(f"   - val2014: {val2014_total}")
        
        # 过滤：只保留实际存在的图像
        print(f"🔍 正在过滤存在的图像...")
        self.data = []
        skipped = 0
        
        for item in all_data:
            img_path = os.path.join(img_dir, item['image'])
            if os.path.exists(img_path):
                self.data.append(item)
                if len(self.data) >= max_samples:
                    break
            else:
                skipped += 1
        
        self.img_dir = img_dir
        self.processor = processor
        
        print(f"✅ 成功加载 {len(self.data)} 个样本")
        print(f"⚠️  跳过 {skipped} 个不存在的图像")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item['image'])
        
        try:
            image = Image.open(img_path).convert('RGB')
            encoding = self.processor(
                images=image,
                text=item['caption'],
                padding="max_length",
                truncation=True,
                max_length=32,
                return_tensors="pt"
            )
            encoding = {k: v.squeeze() for k, v in encoding.items()}
            encoding["labels"] = encoding["input_ids"].clone()
            return encoding
        except Exception as e:
            print(f"⚠️  跳过图像: {img_path}, 错误: {e}")
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}

print("=" * 60)
print("🚀 BLIP-2 微调训练脚本")
print("=" * 60)

print("\n📥 加载模型...")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16
)
model.to("cuda")
print("✅ 模型加载完成")

print("\n📊 加载数据集...")
train_dataset = SmartCOCODataset(
    ann_file="/root/autodl-tmp/datasets/coco/annotations/coco_karpathy_train.json",
    img_dir="/root/autodl-tmp/datasets/coco",
    processor=processor,
    max_samples=5000  # 使用 5000 个样本快速测试
)

print("\n⚙️  配置训练参数...")
training_args = TrainingArguments(
    output_dir="./blip2-coco-ft",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_steps=200,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    dataloader_num_workers=2,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

print("\n🚀 开始训练...")
print("=" * 60)
trainer.train()

print("\n" + "=" * 60)
print("✅ 训练完成！")
print(f"📁 模型保存在: ./blip2-coco-ft")
print("=" * 60)
