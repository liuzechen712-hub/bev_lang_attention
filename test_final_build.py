import sys
sys.path.insert(0, '.')
from projects.mmdet3d_plugin import *
from mmdet3d.models.builder import build_detector
from mmcv.utils import ConfigDict
import copy

def deep_to_plain_dict(obj):
    """递归将 ConfigDict 转换为普通 dict，切断所有 .parent 引用"""
    if isinstance(obj, dict):
        return {k: deep_to_plain_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [deep_to_plain_dict(i) for i in obj]
    else:
        return obj

# 1. 定义基础配置
head_cfg = {
    'type': 'projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head.BEVFormerHead',
    'num_classes': 10, 
    'in_channels': 256, 
    'num_query': 900,
    'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    'loss_cls': {'type': 'FocalLoss', 'use_sigmoid': True, 'gamma': 2.0, 'alpha': 0.25, 'loss_weight': 2.0},
    'loss_bbox': {'type': 'L1Loss', 'loss_weight': 0.25}
}

train_cfg = {
    'pts': {
        'assigner': {
            'type': 'HungarianAssigner3D',
            'cls_cost': {'type': 'FocalLossCost', 'weight': 2.0},
            'reg_cost': {'type': 'BBox3DL1Cost', 'weight': 0.25},
            'iou_cost': {'type': 'IoUCost', 'weight': 0.0},
            'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        }
    }
}

test_cfg = {
    'pts': {'max_per_img': 100}
}

model_cfg = {
    'type': 'BEVFormer',
    'use_grid_mask': True,
    'img_backbone': {
        'type': 'ResNet', 'depth': 50, 'num_stages': 4, 'out_indices': (1, 2, 3),
        'frozen_stages': 1, 'norm_cfg': {'type': 'BN2d', 'requires_grad': False},
        'norm_eval': True, 'style': 'pytorch'
    },
    'img_neck': {
        'type': 'FPN', 'in_channels': [512, 1024, 2048], 'out_channels': 256,
        'start_level': 0, 'add_extra_convs': 'on_output', 'num_outs': 4,
        'relu_before_extra_convs': True
    },
    'pts_bbox_head': head_cfg,
    'train_cfg': train_cfg,
    'test_cfg': test_cfg
}

# 2. 关键步骤：先转为 ConfigDict，再立即转回纯普通字典
#    这一步是为了模拟 tools/train.py 的行为，然后立即“消毒”
cfg_dict = ConfigDict(model_cfg)
plain_cfg = deep_to_plain_dict(cfg_dict)

# 3. 再次转为 ConfigDict（此时内部已无复杂继承链）
final_cfg = ConfigDict(plain_cfg)

print("🔨 Building model with sanitized ConfigDict...")
try:
    model = build_detector(final_cfg)
    print("✅ SUCCESS! Model built.")
    print(f"   Head type: {type(model.pts_bbox_head).__name__}")
except Exception as e:
    print(f"❌ Failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
