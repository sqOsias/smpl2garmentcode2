"""验证固定服装版型从预测体型驱动到 CloSe GT 体型与姿态。

本脚本复用已有 ``drive_garment`` 和 ``evaluate_driven_garment``，不重新生成
纸样或执行 GarmentCode 初始仿真。输入目录需已有 design_sim.obj、boxmesh、
smpl.obj 和 HybrIK smpl.json。
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.getcwd())

from pygarment.meshgen.garment_driver import drive_garment
from work.metric import evaluate_driven_garment, save_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Drive an existing simulated garment to GT shape+pose and evaluate it."
    )
    parser.add_argument('--sample', required=True, help='CloSe sample id, e.g. 10112_9583')
    parser.add_argument('--data_root', required=True, help='Directory containing CloSe-Di NPZ files')
    parser.add_argument('--output_root', required=True, help='AutoGarmentCode/output/CloSe directory')
    parser.add_argument('--gender', default='male', choices=('male', 'female'),
                        help='Gender of the HybrIK/reference body')
    parser.add_argument('--smpl_model_path',
                        default='/root/wyc/code/smpl2garmentcode2/smpl_models')
    parser.add_argument('--sim_config', default=None,
                        help='Override simulation YAML; defaults to design/sim_props.yaml')
    parser.add_argument('--no_intermediate', action='store_true',
                        help='Do not export transition/simulation intermediate OBJ files')
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def main():
    args = parse_args()
    sample_dir = Path(args.output_root) / args.sample
    design_dir = sample_dir / 'design'
    target_npz = require_file(Path(args.data_root) / f'{args.sample}.npz', 'CloSe NPZ')
    garment_obj = require_file(design_dir / 'design_sim.obj', 'simulated garment OBJ')
    boxmesh_obj = require_file(design_dir / 'design_boxmesh.obj', 'boxmesh OBJ')
    body_obj = require_file(sample_dir / 'smpl.obj', 'reference A-pose body OBJ')
    base_smpl_json = require_file(sample_dir / 'hybrik' / 'smpl.json', 'HybrIK SMPL JSON')

    if args.sim_config is None:
        sim_config = design_dir / 'sim_props.yaml'
        if not sim_config.is_file():
            sim_config = Path('assets/Sim_props/default_sim_props.yaml')
    else:
        sim_config = Path(args.sim_config)
    require_file(sim_config, 'simulation config')

    drive_output = design_dir / 'driven_gt_shape'
    metric_output = sample_dir / 'gt_shape_validation'
    drive_output.mkdir(parents=True, exist_ok=True)
    metric_output.mkdir(parents=True, exist_ok=True)

    with np.load(target_npz) as npz:
        target_data = {key: npz[key] for key in npz.files}

    print('=' * 70)
    print(f'GT-shape drive validation: {args.sample}')
    print(f'Garment: {garment_obj}')
    print(f'Drive output: {drive_output}')
    print(f'Metric output: {metric_output}')
    print('=' * 70)

    driven = drive_garment(
        garment_obj_path=str(garment_obj),
        boxmesh_obj_path=str(boxmesh_obj),
        body_obj_path=str(body_obj),
        smpl_model_path=args.smpl_model_path,
        base_smpl_json=str(base_smpl_json),
        target_pose_npz=str(target_npz),
        target_data=target_data,
        gender=args.gender,
        output_dir=str(drive_output),
        sim_config_path=str(sim_config),
        save_intermediate=not args.no_intermediate,
        drive_to_gt_shape=True,
    )

    geometry = evaluate_driven_garment(
        driven_verts_m=driven['driven_verts_m'],
        driven_faces=driven['garment_faces'],
        pred_target_body_verts_m=driven['target_body_v_m'],
        gt_target_body_verts_m=driven['gt_target_body_v_m'],
        gt_template_verts_m=driven['gt_template_v_m'],
        gt_data=target_data,
        torso_mask=driven['smpl_torso_mask'],
        output_dir=str(metric_output),
    )

    summary = {
        'sample_name': args.sample,
        'validation_mode': 'fixed_pattern_gt_shape_and_pose',
        'chamfer_distance': geometry['cd_cm'],
        'f_score': geometry['fscore_10mm'],
        'fscores': geometry['fscores'],
        'scan_alignment': geometry['scan_alignment'],
    }
    save_metrics(summary, str(metric_output))

    print('\nGT-shape validation result')
    print(f"  CD: {geometry['cd_cm']:.4f} cm")
    print(f"  F@10mm: {geometry['fscore_10mm']:.4f}")
    for threshold, values in geometry['fscores'].items():
        print(f"  F@{threshold}mm: {values['fscore']:.4f}")


if __name__ == '__main__':
    main()
