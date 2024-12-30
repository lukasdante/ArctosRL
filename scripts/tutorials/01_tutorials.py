# Import Isaac Sim AppLauncher
import argparse
from omni.isaac.lab.app import AppLauncher
# Create argument parser
parser = argparse.ArgumentParser(description='Tutorial on interacting with an articulation.')
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import necessary libraries
import torch

import omni.isaac.core.utils.prims as prim_utils # type: ignore

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext

# Predefined cartpole configuration
from omni.isaac.lab_assets import CARTPOLE_CFG


def design_scene() -> tuple[dict, list[list[float]]]:
    # Create a non-interactive ground plane prim
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func('/World/defaultGroundPlane', cfg)
    # Create a light prim
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func('/World/Light', cfg)

    # Create prims for cartpoles with certain origins
    origins = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f'/World/Origin{i}', 'Xform', translation=origin)

    # Clone the articulation configuration
    cartpole_cfg = CARTPOLE_CFG.copy()
    # Create a prim path for the articulation
    cartpole_cfg.prim_path = '/World/Origin.*/Robot'
    # Create the articulation
    cartpole = Articulation(cfg=cartpole_cfg)

    # return scene entities
    scene_entities = {'cartpole': cartpole}
    return scene_entities, origins

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    # Get the cartpole scene entity
    robot = entities['cartpole']
    # Get the physics time step
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        # Reset the simulation
        if count % 500 == 0:
            count = 0
            # Clone the default root state
            root_state = robot.data.default_root_state.clone()
            # Since the prims are referenced from the simulation world frame
            # you must displace each prim with an origin value such that
            # it doesn't spawn in (0, 0, 0)
            root_state[:, :3] += origins
            # write the root state to the sim
            robot.write_root_state_to_sim(root_state)
            # Clone the default joint state
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            # Add some noise to joint position
            joint_pos += torch.rand_like(joint_pos) * 0.1
            # write joint state to sim
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # reset
            robot.reset()
        # create some joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # set the joint efforts (can also be joint position or velocity)
        robot.set_joint_effort_target(efforts)
        # write the data to sim (external forces)
        robot.write_data_to_sim()
        # step the simulation
        sim.step()
        count += 1
        # update the data buffers
        robot.update(sim_dt)

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    sim.reset()

    run_simulator(sim, scene_entities, scene_origins)

if __name__=='__main__':
    main()
    simulation_app.close()