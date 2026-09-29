import re

# 读取文件
with open('lavis/models/blip2_models/blip2_pretrain.py', 'r') as f:
    content = f.read()

# 查找并替换 init_vision_encoder 调用
old_pattern = r'self\.visual_encoder, self\.ln_vision = self\.init_vision_encoder\(\s*vit_model,\s*img_size\s*\)'

new_code = '''self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, 
            img_size, 
            drop_path_rate=0.0,
            use_grad_checkpoint=use_grad_checkpoint,
            precision="fp16"
        )'''

content = re.sub(old_pattern, new_code, content, flags=re.DOTALL)

# 写回文件
with open('lavis/models/blip2_models/blip2_pretrain.py', 'w') as f:
    f.write(content)

print("✅ init_vision_encoder 调用已修复！")
