import torch
import numpy as np
import matplotlib.pyplot as plt
from mmcv import Config
from mmcv.parallel import DataContainer as DC
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from projects.mmdet3d_plugin import *

def visualize_features(model, data_loader, save_path='feature_activation.png'):
    model.eval()
    with torch.no_grad():
        data = next(iter(data_loader))
        img_data = data['img']
        while isinstance(img_data, DC): img_data = img_data.data
        if isinstance(img_data, list): img_data = img_data[0]
        while img_data.dim() > 5: img_data = img_data.squeeze(0)
        while img_data.dim() < 5: img_data = img_data.unsqueeze(0)
        img_gpu = img_data.cuda()
        
        metas_data = data['img_metas']
        while isinstance(metas_data, DC): metas_data = metas_data.data
        if isinstance(metas_data, list) and len(metas_data) > 0 and isinstance(metas_data[0], list):
            metas_data = metas_data[0]

        feats = model.extract_feat(img=img_gpu, img_metas=metas_data)
        last_feat = feats[-1]
        if isinstance(last_feat, list): last_feat = last_feat[-1]
        if last_feat.dim() == 5: last_feat = last_feat.mean(dim=1)
        activation_map = last_feat[0].mean(dim=0).squeeze().cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(activation_map, cmap='hot', interpolation='nearest')
        plt.title('Deep Feature Activation Map')
        plt.axis('off')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

if __name__ == '__main__':
    cfg = Config.fromfile('projects/configs/bevformer/bevformer_standalone_test.py')
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
    model = build_model(cfg.model)
    checkpoint = torch.load('work_dirs/bevformer_enhanced/epoch_5.pth', map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.cuda()
    visualize_features(model, data_loader)
