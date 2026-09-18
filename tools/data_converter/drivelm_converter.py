import os
import json
import pickle
import argparse
from nuscenes.nuscenes import NuScenes

def convert_drivelm_to_bevformer(root_path, json_path, version='v1.0-trainval', out_name='infos_train.pkl'):
    print(f"🔍 Loading NuScenes meta from: {os.path.join(root_path, version)}")
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    
    print(f"📖 Loading DriveLM JSON from: {json_path}")
    with open(json_path, 'r') as f:
        drivelm_data = json.load(f)
    
    infos = []
    scene_tokens = list(drivelm_data.keys())
    print(f"🔄 Converting {len(scene_tokens)} scenes...")
    
    for i, scene_token in enumerate(scene_tokens):
        if i % 50 == 0:
            print(f"   Processing scene {i+1}/{len(scene_tokens)}: {scene_token[:8]}...")
            
        scene_rec = nusc.get('scene', scene_token)
        sample_token = scene_rec['first_sample_token']
        
        while sample_token:
            sample_rec = nusc.get('sample', sample_token)
            
            info = {
                'token': sample_token,
                'timestamp': sample_rec['timestamp'],
                'scene_token': scene_token,
                'cams': {},
                'lidar2ego_translation': [0.0, 0.0, 0.0],
                'lidar2ego_rotation': [1.0, 0.0, 0.0, 0.0],
                'gt_boxes': [],
                'gt_names': [],
            }
            
            # 1. 提取6个相机的标定与位姿信息，并替换为DriveLM实际路径
            for cam_type in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                sd_token = sample_rec['data'][cam_type]
                sd_rec = nusc.get('sample_data', sd_token)
                cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
                ep_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
                
                original_filename = sd_rec['filename']
                drive_lm_filename = f"nuscenes/samples/{cam_type}/{os.path.basename(original_filename)}"
                
                info['cams'][cam_type] = {
                    'data_path': drive_lm_filename,
                    'type': cam_type,
                    'extrinsics': {
                        'rotation': cs_rec['rotation'],
                        'translation': cs_rec['translation'],
                    },
                    'intrinsics': cs_rec['camera_intrinsic'],
                    'ego2global_rotation': ep_rec['rotation'],
                    'ego2global_translation': ep_rec['translation'],
                    'timestamp': sd_rec['timestamp'],
                }
            
            # 2. 提取3D标注框 (兼容新旧版SDK)
            if sample_rec['anns']:
                for ann_token in sample_rec['anns']:
                    ann_rec = nusc.get('sample_annotation', ann_token)
                    
                    if 'category_name' in ann_rec:
                        cat_name = ann_rec['category_name']
                    elif 'category_token' in ann_rec:
                        cat_rec = nusc.get('category', ann_rec['category_token'])
                        cat_name = cat_rec['name']
                    else:
                        cat_name = "unknown"
                        
                    info['gt_boxes'].append(ann_rec['size'] + ann_rec['translation'] + ann_rec['rotation'])
                    info['gt_names'].append(cat_name)
            
            # 3. 附加DriveLM QA对占位
            info['drivelm_qa'] = [] 
            
            infos.append(info)
            sample_token = sample_rec['next']
    
    output_path = os.path.join(root_path, out_name)
    data = {'infos': infos}
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ Successfully saved {len(infos)} samples to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-path', type=str, default='./data/nuscenes')
    parser.add_argument('--json-path', type=str, default='./data/nuscenes/v1_1_train_nus.json')
    parser.add_argument('--out-name', type=str, default='infos_train.pkl')
    args = parser.parse_args()
    
    convert_drivelm_to_bevformer(args.root_path, args.json_path, out_name=args.out_name)
