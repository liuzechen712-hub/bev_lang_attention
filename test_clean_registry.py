import sys
sys.path.insert(0, '.')

# 1. 清除所有可能污染的模块缓存
mods_to_clear = [k for k in sys.modules.keys() if 'mmdet3d_plugin' in k or 'bevformer' in k]
for mod in mods_to_clear:
    del sys.modules[mod]

# 2. 导入核心构建器
from mmdet3d.models.builder import DETECTORS, HEADS

# 3. 检查注册表当前的 parent 状态
print(f"🔍 Inspecting DETECTORS registry...")
print(f"   Type: {type(DETECTORS)}")
print(f"   Has parent attr: {hasattr(DETECTORS, 'parent')}")
if hasattr(DETECTORS, 'parent'):
    print(f"   Parent value: {DETECTORS.parent}")
    print(f"   Parent type: {type(DETECTORS.parent)}")

# 4. 手动注册 BEVFormer（不使用 force，先检查是否存在）
from projects.mmdet3d_plugin.bevformer.detectors.bevformer import BEVFormer

if 'BEVFormer' not in DETECTORS.module_dict:
    print("   Registering BEVFormer...")
    DETECTORS.register_module(module=BEVFormer)
else:
    print("   BEVFormer already registered.")

# 5. 再次检查 parent 状态
if hasattr(DETECTORS, 'parent'):
    print(f"   Parent after register: {DETECTORS.parent}")

# 6. 尝试获取类
print("🔨 Trying to get BEVFormer from registry...")
try:
    cls = DETECTORS.get('BEVFormer')
    print(f"✅ Successfully retrieved: {cls}")
except AttributeError as e:
    print(f"❌ Failed to get: {e}")
    print("   This confirms the Registry internal structure is broken.")
