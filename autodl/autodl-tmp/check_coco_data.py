#!/usr/bin/env python3
# check_coco_data.py — 通用版：自动探测并加载 COCO caption 数据集，打印诊断与样例

import os, sys, traceback
print("Python:", sys.executable)
print("PWD:", os.getcwd())

# 先打印 lavis 安装信息
try:
    import lavis
    print("LAVIS module:", lavis.__file__)
except Exception as e:
    print("无法 import lavis:", e)
    traceback.print_exc()
    sys.exit(1)

# 显示 builders 目录下有哪些文件，便于判断导入名
builders_dir = os.path.join(os.path.dirname(lavis.__file__), "datasets", "builders")
print("\n--- builders dir:", builders_dir)
if os.path.isdir(builders_dir):
    print("builders 内容：")
    for f in sorted(os.listdir(builders_dir)):
        print("  ", f)
else:
    print("找不到 builders 目录，路径可能不对。")

# 你的 COCO 数据路径，按你当前环境修改
ANNOTATIONS = "/root/autodl-tmp/datasets/coco/annotations/annotations/captions_val2014.json"
IMAGES_DIR = "/root/autodl-tmp/datasets/coco/val2014"

print("\n使用的数据路径：")
print("  ANNOTATIONS:", ANNOTATIONS)
print("  IMAGES_DIR:", IMAGES_DIR)
print("  注：请确保这两个路径存在且可读。")
print("  ANNOTATIONS exists?", os.path.exists(ANNOTATIONS))
print("  IMAGES_DIR exists?", os.path.exists(IMAGES_DIR))

# 尝试多种导入方式 —— 优先使用 builder，如果没有则回退到 load_dataset
success = False
errors = []

# Try 1: 尝试常见 builder 导入名
builder_candidates = [
    "lavis.datasets.builders.coco_caption_builder",
    "lavis.datasets.builders.caption_datasets",
    "lavis.datasets.builders.coco_caption",
    "lavis.datasets.builders.coco_caption_builder_v1"
]

for modname in builder_candidates:
    try:
        mod = __import__(modname, fromlist=["*"])
        print(f"\n尝试导入 builder 模块：{modname} —— 成功")
        # 在不同实现里类名可能不同，常见名：
        class_names = ["COCODatasetBuilder", "COCOCaptionBuilder", "CocoCaptionBuilder", "COCODataset"]
        builder_obj = None
        for cname in class_names:
            if hasattr(mod, cname):
                BuilderClass = getattr(mod, cname)
                print("  找到 builder 类名：", cname)
                try:
                    b = BuilderClass()
                except Exception:
                    b = None
                builder_obj = b
                break
        if builder_obj is not None:
            # 设置 build_info（不同版本字段名可能不同，但这是常见格式）
            try:
                builder_obj.build_info = {
                    "annotations": [ANNOTATIONS],
                    "images": IMAGES_DIR
                }
                print("  已设置 builder.build_info")
                dataset_dict = builder_obj.build_datasets()
                print("  build_datasets() 返回 keys:", list(dataset_dict.keys()))
                # 选择 val split 或第一个键
                key = "val" if "val" in dataset_dict else (list(dataset_dict.keys())[0])
                dataset = dataset_dict[key]
                print(f"✅ 从 builder 成功得到数据集：键 = {key}, 样本数 = {len(dataset)}")
                # 打印第一个样本（若存在）
                if len(dataset) > 0:
                    s = dataset[0]
                    print("示例样本 keys:", list(s.keys()))
                    print("示例 image:", s.get("image"))
                    print("示例 caption:", s.get("caption"))
                success = True
                break
            except Exception as e:
                print("  调用 builder 失败：", e)
                traceback.print_exc()
                errors.append(("builder_call", modname, str(e)))
        else:
            print("  找不到已知的 builder 类名，跳过此模块。")
    except Exception as e:
        print(f"\n尝试导入 {modname} 失败：{e}")

if not success:
    # Try 2: 回退到 builders 包中的 load_dataset（你之前试过，但参数签名不同）
    try:
        from lavis.datasets.builders import load_dataset
        print("\n尝试使用 lavis.datasets.builders.load_dataset(...)")
        print("load_dataset callable:", callable(load_dataset))
        # 尝试不同调用方式：先无参数调用（部分版本返回 dict）
        try:
            print("  尝试：dataset_dict = load_dataset('coco_caption')")
            dataset_dict = load_dataset("coco_caption")
            print("  返回类型：", type(dataset_dict))
            if isinstance(dataset_dict, dict):
                print("  返回 keys:", list(dataset_dict.keys()))
                key = "val" if "val" in dataset_dict else (list(dataset_dict.keys())[0])
                dataset = dataset_dict[key]
                print(f"✅ load_dataset 成功得到数据集：键 = {key}, 样本数 = {len(dataset)}")
                if len(dataset) > 0:
                    s = dataset[0]
                    print("示例样本 keys:", list(s.keys()))
                    print("示例 image:", s.get("image"))
                    print("示例 caption:", s.get("caption"))
                success = True
        except TypeError as te:
            print("  load_dataset(...) 抛出 TypeError（签名可能不支持该调用），错误：", te)
            errors.append(("load_dataset_typeerror", str(te)))
        except Exception as e:
            print("  load_dataset 调用失败：", e)
            traceback.print_exc()
            errors.append(("load_dataset_other", str(e)))

    except Exception as e:
        print("无法从 lavis.datasets.builders 导入 load_dataset:", e)
        traceback.print_exc()
        errors.append(("import_load_dataset", str(e)))

if not success:
    print("\n❗ 未能自动加载数据集。以下为建议的下一步排查：")
    print("1) 列出 builders 目录内容，确认有哪些模块名（脚本顶部已经打印）。")
    print("2) 把 builders 目录列出并把输出粘贴给我：")
    print("   ls -la", builders_dir)
    print("3) 如果你看到某个模块名很像 coco-caption builder（例如文件名包含 'coco' 或 'caption'），把该文件名发来，我帮你写针对性的导入代码。")
    print("4) 如果 load_dataset 返回了 dict 但没有 val 键，请把 dict.keys() 的输出粘贴给我。")
    print("\n错误跟踪（部分）：")
    for e in errors[:10]:
        print(" -", e)
    sys.exit(2)

# 成功的话在这里结束
print("\n脚本执行完毕（成功）。")
