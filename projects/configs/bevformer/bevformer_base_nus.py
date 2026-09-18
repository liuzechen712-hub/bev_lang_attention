_base_ = [
    '../_base_/datasets/nus-3d.py',
    '../_base_/default_runtime.py'
]

# ==================== 模型配置 (严格对齐 BEVFormer.__init__ 签名) ====================
model = dict(
    type='BEVFormer',
    use_grid_mask=True,
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
    pts_voxel_layer=dict(
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_num_points=10, voxel_size=[0.2, 0.2, 8], max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1440, 1440],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    # ✅ point_cloud_range 仅存在于 Head 和 Assigner 内部
    pts_bbox_head=dict(
        type='projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head.BEVFormerHead',
        num_classes=10,
        in_channels=256,
        num_query=900,
        num_reg_fcs=2,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        transformer=dict(
            type='PerceptionTransformer',
            embed_dims=256,
            num_heads=8,
            num_layers=6,
            num_levels=4,
            num_cams=6,
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            dropout=0.1,
            batch_first=True),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=128,
            row_num_embed=50,
            col_num_embed=50),
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]))),
    test_cfg=dict(pts=dict(max_per_img=100)))

# ==================== 数据配置 ====================
dataset_type = 'CustomNuScenesDatasetV2'
data_root = './data/nuscenes'

file_client_args = dict(backend='disk')
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(type='ObjectNameFilter', classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']),
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
        classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=False,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/infos_val.pkl',
        pipeline=test_pipeline,
        classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/infos_val.pkl',
        pipeline=test_pipeline,
        classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=True,
        box_type_3d='LiDAR'))

# ==================== 训练策略 ====================
optimizer = dict(type='AdamW', lr=2e-4, paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=500, warmup_ratio=1.0 / 3, min_lr_ratio=1e-5)
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='SetEpochInfoHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
fp16 = dict(loss_scale=512.)
find_unused_parameters = True
