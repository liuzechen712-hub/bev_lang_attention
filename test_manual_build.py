import sys
sys.path.insert(0, '.')

# 1. 初始化插件注册
from projects.mmdet3d_plugin import *

# 2. 手动定义最小化配置字典（完全避开 MMCV Config 类）
model_cfg = {
    'type': 'BEVFormer',
    'use_grid_mask': True,
    'img_backbone': {
        'type': 'ResNet',
        'depth': 50,
        'num_stages': 4,
        'out_indices': (1, 2, 3),
        'frozen_stages': 1,
        'norm_cfg': {'type': 'BN2d', 'requires_grad': False},
        'norm_eval': True,
        'style': 'pytorch'
    },
    'img_neck': {
        'type': 'FPN',
        'in_channels': [512, 1024, 2048],
        'out_channels': 256,
        'start_level': 0,
        'add_extra_convs': 'on_output',
        'num_outs': 4,
        'relu_before_extra_convs': True
    },
    'pts_bbox_head': {
        'type': 'projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head.BEVFormerHead',
        'num_classes': 10,
        'in_channels': 256,
        'num_query': 900,
        'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        'loss_cls': {'type': 'FocalLoss', 'use_sigmoid': True, 'gamma': 2.0, 'alpha': 0.25, 'loss_weight': 2.0},
        'loss_bbox': {'type': 'L1Loss', 'loss_weight': 0.25}
    },
    'train_cfg': {'pts': {'assigner': {'type': 'HungarianAssigner3D'}}},
    'test_cfg': {'pts': {'max_per_img': 100}}
}

# 3. 手动构建模型
from mmdet3d.models.builder import build_detector

print("🔨 Manually building BEVFormer model...")
try:
    # 注意：build_detector 期望的是 ConfigDict，但我们传入普通 dict 看看是否触发同样的错误
    from mmcv.utils import ConfigDict
    cfg_dict = ConfigDict(model_cfg)
    model = build_detector(cfg_dict)
    print("✅ Model built successfully!")
    print(f"   Model type: {type(model).__name__}")
except Exception as e:
    print(f"❌ Manual build failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
