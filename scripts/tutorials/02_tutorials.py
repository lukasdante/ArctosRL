import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description='Tutorial for interactive scenes.')
parser.add_argument('--num_envs', type=int, default=2, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext, SimulationCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_assets import CARTPOLE_CFG

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path='/World/defaultGroundPlane', spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(prim_path='/World/Light', spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    cartpole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot: Articulation = scene['cartpole']
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            scene.reset()
            print("[INFO]: Resetting the state.")
        efforts = torch.rand_like(robot.data.joint_pos) * 5.0
        robot.set_joint_effort_target(efforts)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

def main():
    sim_cfg = SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    scene_cfg = CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()