_base_ = ['./bevformer_base_nus.py']  # 继承我们刚才修好的主配置

# 覆盖数据路径确保指向正确
data = dict(
    train=dict(ann_file='./data/nuscenes/infos_train.pkl'),
    val=dict(ann_file='./data/nuscenes/infos_val.pkl'),
    test=dict(ann_file='./data/nuscenes/infos_val.pkl')
)

# 强制缩短训练周期用于测试
runner = dict(max_epochs=1)
evaluation = dict(interval=999999)
