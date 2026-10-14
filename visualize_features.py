import torch
import numpy as np
import matplotlib.pyplot as plt
from mmcv import Config
from mmcv.parallel import DataContainer as DC
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from projects.mmdet3d_plugin import * # 注册插件

def visualize_features(model, data_loader, save_path='feature_activation.png'):
    model.eval()
    with torch.no_grad():
        # 获取第一个 batch 的数据
        data = next(iter(data_loader))
        
        # ================= 1. 暴力解包并修正 img 维度 =================
        img_data = data['img']
        # 剥离 DataContainer
        while isinstance(img_data, DC):
            img_data = img_data.data
        # 如果是 list (代表 queue frames)，取第一个元素 (当前帧)
        if isinstance(img_data, list):
            img_data = img_data[0]
            
        # 暴力修正维度到标准的 5D: [Batch, Num_Cams, Channels, H, W]
        # 即 [1, 6, 3, 928, 1600]
        while img_data.dim() > 5:
            img_data = img_data.squeeze(0) # 挤掉多余的 1
        while img_data.dim() < 5:
            img_data = img_data.unsqueeze(0) # 补齐缺失的维度
            
        img_gpu = img_data.cuda()
        print(f"🔍 [DEBUG] 最终 img_gpu shape: {img_gpu.shape}")
        # ===============================================================
        
        # ================= 2. 解包 img_metas =================
        metas_data = data['img_metas']
        while isinstance(metas_data, DC):
            metas_data = metas_data.data
        # 确保是 list of dicts
        if isinstance(metas_data, list) and len(metas_data) > 0 and isinstance(metas_data[0], list):
            metas_data = metas_data[0]
        # =====================================================

        # 3. 提取图像特征 (backbone + neck)
        feats = model.extract_feat(img=img_gpu, img_metas=metas_data)
        
        # 4. 获取最后一层特征
        last_feat = feats[-1]
        if isinstance(last_feat, list):
            last_feat = last_feat[-1]
            
        # 处理可能的多视角维度 (如果包含 Num_Cams 维度，先求均值)
        if last_feat.dim() == 5: # [B, N, C, H, W]
            last_feat = last_feat.mean(dim=1) # -> [B, C, H, W]
            
        # 取 batch 的第一个样本，并在通道维度求均值，得到 2D 空间激活图
        activation_map = last_feat[0].mean(dim=0).squeeze().cpu().numpy()
        
        # 5. 绘制热力图
        plt.figure(figsize=(10, 8))
        im = plt.imshow(activation_map, cmap='hot', interpolation='nearest')
        plt.title('Deep Feature Activation Map (Last Backbone Stage)', fontsize=14)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 特征激活图已成功保存至: {save_path}")

if __name__ == '__main__':
    print("🔍 正在加载配置和数据...")
    cfg = Config.fromfile('projects/configs/bevformer/bevformer_standalone_test.py')
    
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset, 
        samples_per_gpu=1, 
        workers_per_gpu=0, 
        dist=False, 
        shuffle=False
    )
    
    print("🔍 正在构建模型并加载权重...")
    model = build_model(cfg.model)
    
    checkpoint_path = 'work_dirs/bevformer_enhanced/epoch_5.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.cuda()
    
    print("🚀 开始提取特征并可视化...")
    visualize_features(model, data_loader)
