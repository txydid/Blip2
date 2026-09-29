with open('lavis/models/blip2_models/blip2_pretrain.py', 'r') as f:
    content = f.read()

# 替换 init_Qformer 调用
# 将 qformer_model 改为 self.visual_encoder.num_features (视觉编码器的输出维度)
old_code = '''self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, qformer_model
        )'''

new_code = '''self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, 
            self.visual_encoder.num_features  # vision_width: 视觉编码器输出维度
        )'''

content = content.replace(old_code, new_code)

with open('lavis/models/blip2_models/blip2_pretrain.py', 'w') as f:
    f.write(content)

print("✅ 已修复 init_Qformer 调用参数")
