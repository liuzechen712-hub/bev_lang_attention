plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

model = dict(
    type='BEVFormer',
    use_grid_mask=True,
    # 暂时移除 pts_voxel_layer 等点云分支，只保留图像分支测试
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    # 极简 Head 配置，移除复杂的 transformer 嵌套
    pts_bbox_head=dict(
        type='projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head.BEVFormerHead',
        num_classes=10,
        in_channels=256,
        num_query=900,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25)),
    train_cfg=dict(pts=dict(assigner=dict(type='HungarianAssigner3D'))),
    test_cfg=dict(pts=dict(max_per_img=100)))

dataset_type = 'CustomNuScenesDatasetV2'
data_root = './data/nuscenes'
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='Collect3D', keys=['img'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/infos_train.pkl',
        pipeline=train_pipeline,
        classes=['car'],
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=False,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/infos_val.pkl',
        pipeline=[],
        classes=['car'],
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=True,
        box_type_3d='LiDAR'))

optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='Step', step=[10])
total_epochs = 1
evaluation = dict(interval=999999)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='SetEpochInfoHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
fp16 = dict(loss_scale=512.)
find_unused_parameters = True
