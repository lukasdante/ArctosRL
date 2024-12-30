import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Manager-based base environment tutorial.")
parser.add_argument('--num_envs', type=int, default=2, help='Number of environments.')
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import torch

import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleSceneCfg


@configclass
class ActionsCfg:
    joint_efforts = mdp.JointEffortActionCfg(asset_name='robot', joint_names=['slider_to_cart'], scale=5.0)
    
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    # on startup
    add_pole_mass = EventTerm(func=mdp.randomize_rigid_body_mass,
                              mode='startup',
                              params={
                                  'asset_cfg': SceneEntityCfg('robot', body_names=['pole']),
                                  'mass_distribution_params': (0.1, 0.5),
                                  'operation': 'add',
                              },
                              )
    
    # on reset
    reset_cart_position = EventTerm(func=mdp.reset_joints_by_offset,
                                    mode='reset',
                                    params={
                                        'asset_cfg': SceneEntityCfg('robot', joint_names=['slider_to_cart']),
                                        'position_range': (-1.0, 1.0),
                                        'velocity_range': (-0.1, 0.1)
                                    },
                                    )
    
    reset_pole_position = EventTerm(func=mdp.reset_joints_by_offset,
                                    mode='reset',
                                    params={
                                        'asset_cfg': SceneEntityCfg('robot', joint_names=['cart_to_pole']),
                                        'position_range': (-0.125 * math.pi, 0.125 * math.pi),
                                        "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
                                    },
                                    )

@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    scene = CartpoleSceneCfg(num_envs=1024, env_spacing=2.5)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz

def main():
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedEnv(env_cfg)

    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            if count % 300 == 0:
                count = 0
                env.reset()
                print("[INFO]: Resetting the environment...")
            joint_efforts = torch.randn_like(env.action_manager.action)
            obs, _ = env.step(joint_efforts)
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            count += 1

    env.close()

if __name__ == '__main__':
    main()
    simulation_app.close()