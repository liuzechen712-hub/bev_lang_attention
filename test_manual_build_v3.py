import sys
sys.path.insert(0, '.')
from projects.mmdet3d_plugin import *
from mmdet3d.models.builder import build_detector
from mmcv.utils import ConfigDict

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

# 2. 组装模型配置
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

# 3. 关键步骤：将整个 model_cfg 转换为 ConfigDict
#    这会自动递归地将所有嵌套 dict 也转换为 ConfigDict
cfg_dict = ConfigDict(model_cfg)

print("🔨 Building model with fully nested ConfigDict...")
try:
    model = build_detector(cfg_dict)
    print("✅ SUCCESS! Model built.")
    print(f"   Head type: {type(model.pts_bbox_head).__name__}")
except Exception as e:
    print(f"❌ Failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
