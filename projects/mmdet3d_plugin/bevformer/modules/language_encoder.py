import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class LanguageEncoder(nn.Module):
    """
    语言语义解析模块 (Language Semantic Parsing Module)
    功能：将自然语言驾驶指令编码为高维语义特征，并投影到与 BEV 特征相同的维度空间。
    """
    def __init__(self, model_name='distilbert-base-uncased', bev_embed_dims=256):
        super().__init__()
        
        # 1. 加载轻量级预训练语言模型的 Tokenizer 和 Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModel.from_pretrained(model_name)
        
        # 获取语言模型输出的隐藏层维度 (DistilBERT 是 768)
        lang_hidden_size = self.text_model.config.hidden_size
        
        # 2. 语义-空间映射机制 (Semantic-Spatial Mapping)
        self.semantic_proj = nn.Sequential(
            nn.Linear(lang_hidden_size, bev_embed_dims),
            nn.LayerNorm(bev_embed_dims),
            nn.ReLU(inplace=True)
        )
        
        # 3. 迁移学习策略：冻结预训练语言模型的参数
        for param in self.text_model.parameters():
            param.requires_grad = False

    def forward(self, instructions):
        """
        Args:
            instructions (list of str): 驾驶指令列表
        Returns:
            lang_feats (torch.Tensor): Shape [B, bev_embed_dims]
        """
        inputs = self.tokenizer(
            instructions, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=64
        )
        
        device = next(self.text_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            
        cls_feat = outputs.last_hidden_state[:, 0, :] 
        lang_feats = self.semantic_proj(cls_feat)
        
        return lang_feats
