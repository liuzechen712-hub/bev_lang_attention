# ==================== Complete BEVFormer Plugin Registration ====================

# 1. Detectors
from mmdet3d.models.builder import DETECTORS as MMDet3D_DETECTORS
from mmdet3d.models.builder import HEADS as MMDet3D_HEADS

from .bevformer.detectors.bevformer import BEVFormer
from .bevformer.detectors.bevformerV2 import BEVFormerV2
MMDet3D_DETECTORS.register_module(module=BEVFormer, force=True)
MMDet3D_DETECTORS.register_module(module=BEVFormerV2, force=True)

# 2. Heads
from .bevformer.dense_heads.bevformer_head import BEVFormerHead, BEVFormerHead_GroupDETR
MMDet3D_HEADS.register_module(module=BEVFormerHead, force=True)
MMDet3D_HEADS.register_module(module=BEVFormerHead_GroupDETR, force=True)

# 3. BBox Coders
from mmdet.core.bbox.builder import BBOX_CODERS
from .core.bbox.coders.nms_free_coder import NMSFreeCoder
BBOX_CODERS.register_module(module=NMSFreeCoder, force=True)

# 4. Assigners
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from .core.bbox.assigners.hungarian_assigner_3d import HungarianAssigner3D
BBOX_ASSIGNERS.register_module(module=HungarianAssigner3D, force=True)

# 5. Match Costs
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from .core.bbox.match_costs.match_cost import BBox3DL1Cost
MATCH_COST.register_module(module=BBox3DL1Cost, force=True)

# 6. Custom Pipelines (only register what's NOT built-in)
from mmdet3d.datasets.builder import PIPELINES as MMDET3D_PIPELINES

from .datasets.pipelines.transform_3d import (
    PadMultiViewImage,
    NormalizeMultiviewImage,
    PhotoMetricDistortionMultiViewImage,
    CustomCollect3D,
    RandomScaleImageMultiViewImage,
)
from .datasets.pipelines.formating import CustomDefaultFormatBundle3D

for cls in [PadMultiViewImage, NormalizeMultiviewImage,
            PhotoMetricDistortionMultiViewImage, CustomCollect3D,
            RandomScaleImageMultiViewImage, CustomDefaultFormatBundle3D]:
    MMDET3D_PIPELINES.register_module(module=cls, force=True)

print("✅ All custom pipelines registered.")

# 7. Datasets
try:
    from .datasets.nuscenes_dataset_v2 import CustomNuScenesDatasetV2
    from mmdet3d.datasets.builder import DATASETS as MMDET3D_DATASETS
    MMDET3D_DATASETS.register_module(module=CustomNuScenesDatasetV2, force=True)
    print("✅ CustomNuScenesDatasetV2 registered.")
except Exception as e:
    print(f"⚠️ Dataset registration warning: {e}")

# 8. Transformer modules
try:
    from .bevformer.modules.transformer import PerceptionTransformer
    from .bevformer.modules.encoder import BEVFormerEncoder, BEVFormerLayer
    from .bevformer.modules.decoder import DetectionTransformerDecoder
    from .bevformer.modules.spatial_cross_attention import SpatialCrossAttention, MSDeformableAttention3D
    from .bevformer.modules.temporal_self_attention import TemporalSelfAttention
    print("✅ All transformer modules imported.")
except Exception as e:
    print(f"⚠️ Transformer module import warning: {e}")

print("✅ BEVFormer Plugin Fully Loaded.")
