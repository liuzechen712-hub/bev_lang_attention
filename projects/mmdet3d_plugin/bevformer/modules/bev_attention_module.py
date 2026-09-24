import torch
import torch.nn as nn

class BEVLanguageAttention(nn.Module):
    """
    BEV空间注意力生成模块 (BEV Spatial Attention Generator)
    功能：基于语言语义特征，动态生成 BEV 空间的注意力掩码，并增强 BEV 特征。
    """
    def __init__(self, bev_dim=256, lang_dim=256, bev_h=200, bev_w=200):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        
        # 1. 空间感知先验机制 (Spatial Perception Prior)
        # 引入可学习的空间位置编码，帮助模型感知 BEV 空间的绝对位置
        self.spatial_prior = nn.Parameter(torch.randn(1, bev_dim, bev_h, bev_w) * 0.02)
        
        # 2. 动态空间注意力生成器 (Dynamic Spatial Attention Generator)
        # 将一维语言特征 [B, C_lang] 映射为二维空间权重 [B, H*W]
        self.attn_generator = nn.Sequential(
            nn.Linear(lang_dim, bev_dim),
            nn.LayerNorm(bev_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bev_dim, bev_h * bev_w)
        )
        
        # 3. 边界平滑策略 (Boundary Smoothing)
        # 使用 3x3 卷积对生成的掩码进行平滑，避免边界过于生硬
        self.smooth_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

    def forward(self, bev_features, lang_feats):
        """
        前向传播：生成注意力掩码并增强 BEV 特征。
        
        Args:
            bev_features (torch.Tensor): 原始 BEV 特征，Shape: [B, C, H, W]
            lang_feats (torch.Tensor): 语言语义特征，Shape: [B, C_lang]
            
        Returns:
            attention_mask (torch.Tensor): BEV 注意力掩码，Shape: [B, 1, H, W] (值域 0~1)
            enhanced_bev (torch.Tensor): 增强后的 BEV 特征，Shape: [B, C, H, W]
        """
        B, C, H, W = bev_features.shape
        
        # 1. 生成动态空间注意力掩码
        # [B, C_lang] -> [B, H*W] -> [B, 1, H, W]
        mask_logits = self.attn_generator(lang_feats).view(B, 1, H, W)
        
        # 融合空间感知先验 (将先验在通道维度求均值，作为空间偏置)
        prior_bias = self.spatial_prior.mean(dim=1, keepdim=True) # [1, 1, H, W]
        mask_logits = mask_logits + prior_bias
        
        # 边界平滑
        mask_logits = self.smooth_conv(mask_logits)
        
        # 归一化到 0-1 之间，作为注意力权重
        attention_mask = torch.sigmoid(mask_logits) 

        # 2. 特征融合机制 (Feature Fusion)
        # 使用残差连接：原始特征 + 原始特征 * 注意力掩码
        # 这样即使掩码全为 0，也不会丢失原始信息
        enhanced_bev = bev_features + (bev_features * attention_mask)
        
        return attention_mask, enhanced_bev
