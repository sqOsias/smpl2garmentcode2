import os
import sys
sys.path.append(os.getcwd()) 
# os.environ['EGL_DEVICE_ID'] = ''
import argparse
from pathlib import Path

import yaml

from assets.garment_programs.meta_garment import MetaGarment
from assets.bodies.body_params import BodyParameters
from pygarment.data_config import Properties

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
def get_args():
	parser = argparse.ArgumentParser(description='Generate garmentcode and run simulation in one step.')
	parser.add_argument('--design_path', type=str, required=True, help='Path to design YAML file.')
	parser.add_argument('--body_path', type=str, default="./assets/bodies/mean_all.yaml", help='Path to custom body YAML file.')
	parser.add_argument('--sim_config', '-s', type=str, default='./assets/Sim_props/default_sim_props.yaml', help='Simulation config YAML path.')
	parser.add_argument('--sim', type=str, default='false', help='Whether to run simulation after generating garment code.')
	# --- pose driven params ---
	parser.add_argument('--target_pose', type=str, default=None,
		help='target pose NPZ file path。')
	parser.add_argument('--base_smpl_json', type=str, default=None,
		help='base SMPL JSON path (including betas, pose)')
	parser.add_argument('--smpl_model_path', type=str,
		default='/root/wyc/code/smpl2garmentcode2/smpl_models',
		help='SMPL model root')
	parser.add_argument('--gender', type=str, default='male',
		help='SMPL gender (male/female)。')
	return parser.parse_args()


if __name__ == '__main__':
	args = get_args()	
	body = BodyParameters(args.body_path)
	with open(args.design_path, 'r') as f:
		design = yaml.safe_load(f)['design']

	piece_name = Path(args.design_path).stem
	piece = MetaGarment(piece_name, body, design)
	pattern = piece.assembly()

	if piece.is_self_intersecting():
		print(f'{piece.name} is self-intersecting')

	sys_props = Properties('./system.json')
	sys_props['output'] = os.path.dirname(args.design_path)
	folder = Path(pattern.serialize(
        sys_props['output'],
        tag='',
        to_subfolder=False,
        with_3d=False,
        with_text=False,
        view_ids=False,
        with_printable=True,
	))
	body.save(folder)

	print(f'Successfully generated garment code for {piece.name} at {folder}')
	if not args.sim or args.sim.lower() == 'false':
		print('Simulation skipped.')
		exit(0)

	spec_path = folder / f'{piece.name}_specification.json'
	if not spec_path.exists():
		raise FileNotFoundError(f'Cannot find generated specification: {spec_path}')

	from pygarment.meshgen.boxmeshgen import BoxMesh
	from pygarment.meshgen.simulation import run_sim
	from pygarment.meshgen.sim_config import PathCofig

	props = Properties(args.sim_config)
	props.set_section_stats('sim', fails={}, sim_time={}, spf={}, fin_frame={}, body_collisions={}, self_collisions={})
	props.set_section_stats('render', render_time={})
	body_yaml = Path(args.body_path)
	body_path_for_sim = str(body_yaml.parent) if args.body_path is not None else None

	garment_name, _, _ = spec_path.stem.rpartition('_')
	paths = PathCofig(
		in_element_path=spec_path.parent,
		out_path=sys_props['output'],
		in_name=garment_name,
		body_name='smpl',
		body_path=body_path_for_sim,
		smpl_body=True,
		add_timestamp=False,
	)

	print(f'Generate box mesh of {garment_name} with resolution {props["sim"]["config"]["resolution_scale"]}...')
	print(f'Garment load: {paths.in_g_spec}')

	garment_box_mesh = BoxMesh(paths.in_g_spec, props['sim']['config']['resolution_scale'])
	garment_box_mesh.load()
	garment_box_mesh.serialize(paths, store_panels=False, uv_config=props['render']['config']['uv_texture'])

	props.serialize(paths.element_sim_props)
	run_sim(
		garment_box_mesh.name,
		props,
		paths,
		save_v_norms=False,
		store_usd=False,
		optimize_storage=False,
		verbose=False,
	)
	props.serialize(paths.element_sim_props)

	# --- pose driven (optional) ---
	if args.target_pose:
		from pygarment.meshgen.garment_driver import drive_garment
		from pygarment.meshgen.render.pythonrender import render_images

		render_props = props['render']

		print(f'\n===== pose driven start =====')
		print(f'  target pose: {args.target_pose}')
		print(f'  base SMPL: {args.base_smpl_json}')
		print(f'  garment OBJ:  {paths.g_sim}')
		print(f'  body OBJ:  {paths.in_body_obj}')

		driven_dir = str(paths.out_el / 'driven')

		driven_result = drive_garment(
			garment_obj_path=str(paths.g_sim),
			boxmesh_obj_path=str(paths.g_box_mesh),
			body_obj_path=str(paths.in_body_obj),
			smpl_model_path=args.smpl_model_path,
			base_smpl_json=args.base_smpl_json,
			target_pose_npz=args.target_pose,
			gender=args.gender,
			output_dir=driven_dir,
			sim_config_path=args.sim_config,
		)

		# render driven garment + target pose body
		# render_images 内部 load_meshes 做 /100 (cm→m)
		# 所以 body_v 需要 cm: target_body_v_m * 100
		# 临时重定向路径，让渲染输出到 driven/ 目录
		original_g_sim = paths.g_sim
		original_out_el = paths.out_el
		original_sim_tag = paths.sim_tag

		paths.g_sim = Path(driven_result['driven_obj'])
		paths.out_el = Path(driven_dir)
		paths.sim_tag = 'driven'

		print(f'\n[render] render images of driven garment with target pose body...')
		render_images(
			paths,
			driven_result['target_body_v_m'] * 100.0,
			driven_result['target_body_f'],
			render_props['config'],
		)

		paths.g_sim = original_g_sim
		paths.out_el = original_out_el
		paths.sim_tag = original_sim_tag
		print(f'===== pose driven complete =====')

	print(f'Success! Generated and simulated garment: {paths.out_el}')
