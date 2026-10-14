import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# 关键修复：导入插件以触发数据集注册
import projects.mmdet3d_plugin

from mmcv import Config
from mmdet3d.datasets import build_dataset
import mmcv

print("🚀 正在加载配置和数据集...")
cfg = Config.fromfile('projects/configs/bevformer/bevformer_standalone_test.py')
dataset = build_dataset(cfg.data.val)

print("📂 正在加载推理结果 (baseline_result.pkl)...")
results = mmcv.load('work_dirs/baseline_result.pkl')

print("📊 正在计算 mAP 和 NDS 指标...")
eval_kwargs = cfg.get('evaluation', {}).copy()
# 移除不需要的参数
for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
    eval_kwargs.pop(key, None)

# 执行评估并打印
res = dataset.evaluate(results, **eval_kwargs)

print("\n" + "="*50)
print(" 基线模型性能指标 (Baseline Metrics):")
print(res)
print("="*50)
