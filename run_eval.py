import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# 1. 注册自定义插件
import projects.mmdet3d_plugin

import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import single_gpu_test
from mmcv.parallel import MMDataParallel

print(" 正在加载配置和数据集...")
cfg = Config.fromfile('projects/configs/bevformer/bevformer_standalone_test.py')
cfg.model.pretrained = None

# 【核心修复】：强制设置正确的 data_root，避免 nuscenes/nuscenes 重复
correct_data_root = '/home/lzc/data/drivelm_nuscenes/'
print(f"✅ 强制设置 data_root 为: {correct_data_root}")

if 'val' in cfg.data:
    cfg.data.train.data_root = correct_data_root
if 'test' in cfg.data:
    cfg.data.test.data_root = correct_data_root

# 构建验证集
dataset = build_dataset(cfg.data.train)

# 调试：打印第一个样本的图片路径，确认拼接是否正确
print("🔍 调试：检查第一个样本的图片路径...")
first_info = dataset.data_infos[0]
# 假设图片路径在 first_info['cams']['CAM_FRONT']['data_path'] 或类似字段
if 'cams' in first_info:
    sample_path = first_info['cams']['CAM_FRONT']['data_path']
    full_path = os.path.join(correct_data_root, sample_path)
    print(f"   pkl 中的相对路径: {sample_path}")
    print(f"   拼接后的完整路径: {full_path}")
    print(f"   文件是否存在: {os.path.exists(full_path)}")

data_loader = build_dataloader(
    dataset, 
    samples_per_gpu=1, 
    workers_per_gpu=2, 
    dist=False, 
    shuffle=False
)

print("🧠 正在构建模型并加载权重...")
model = build_model(cfg.model)
model = MMDataParallel(model.cuda(), device_ids=[0])
model.eval()

# 加载权重并手动过滤不匹配的 key (解决 lang_encoder 报错)
checkpoint = torch.load('work_dirs/bevformer_baseline/epoch_1.pth', map_location='cpu')
state_dict = checkpoint.get('state_dict', checkpoint)

model_dict = model.state_dict()
filtered_state_dict = {}
for k, v in state_dict.items():
    if k in model_dict and model_dict[k].shape == v.shape:
        filtered_state_dict[k] = v

model.load_state_dict(filtered_state_dict, strict=False)

print("🏃 正在运行推理 (这可能需要几分钟)...")
outputs = single_gpu_test(model, data_loader)

print("📊 正在计算 mAP 和 NDS 指标...")
eval_kwargs = cfg.get('evaluation', {}).copy()
for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
    eval_kwargs.pop(key, None)

res = dataset.evaluate(outputs, **eval_kwargs)

print("\n" + "="*50)
print(" 基线模型性能指标 (Baseline Metrics):")
print(res)
print("="*50)
