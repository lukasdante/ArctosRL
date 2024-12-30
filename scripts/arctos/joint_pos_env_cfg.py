# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from omni.isaac.lab.utils import configclass
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
from omni.isaac.lab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets import ARCTOS_CFG  # isort: skip


##
# Environment configuration
##

# We are simply inheriting from the ReachEnvCfg() class prepared by Isaac Lab
# developers. This is a good start but should you wish to adjust the reward terms,
# curriculum, command terms, etc. You can do it here.
# Isaac Lab uses Hydra configuration system, so you can simply adjust everything here

# To proceed with this, you may need to check the tutorial:
# Creating a Manager-Based RL Environment
# This is important because it will serve as the entry point of our gym registry
# The gym registry can be found in __init__.py

# The agents there are copied from Franka Reach environment.
# Abiding convention, I named the environment Isaac-Reach-Arctos-Play-v0.

@configclass
class ArctosReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Switch robot to Arctos robot arm
        self.scene.robot = ARCTOS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Set the initial state position from its local origin to accommodate for the
        # table and to position itself at the edge
        self.scene.robot.init_state.pos = (-0.3, 0.0, 0.8)

        # Create a 0.8 m^3 cuboid that will serve as the table for the environment
        # This is good practice even if it is only reach task
        self.scene.table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.CuboidCfg(size=(0.8, 0.8, 0.8),
                                      rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                      mass_props=sim_utils.MassPropertiesCfg(mass=1000.0), # set the mass to a very high value 
                                      collision_props=sim_utils.CollisionPropertiesCfg(),
                                      visual_material=sim_utils.PreviewSurfaceCfg(metallic=0.2, diffuse_color=(0.05, 0.05, 0.05)),
                                      ),
                                      )
        
        # Create the non-interactive prims 
        self.scene.ground = AssetBaseCfg(prim_path='/World/defaultGroundPlane', spawn=sim_utils.GroundPlaneCfg())
        self.scene.light = AssetBaseCfg(prim_path='/World/light', spawn=sim_utils.DistantLightCfg(intensity=5000.0, color=(0.75, 0.75, 0.75)))
        
        # override rewards, the body_name that is used to track the command is the
        # 'gripper_assembly_outer', you can find it in the USD file
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper_assembly_outer"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper_assembly_outer"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper_assembly_outer"]

        # override actions, I used RelativeJointPositionActionCfg here because I use MKS Servo42D & 57D. You may
        # also use other action term but in my use case I want to use command F4 (relative motion by axis)
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(asset_name='robot', joint_names=['.*'], scale=1.0)
        
        # override command generator body
        # Adjust the ranges for the target translation and orientation,
        # you can be creative here but make sure you are within the work envelope
        self.commands.ee_pose.body_name = "gripper_assembly_outer"
        self.commands.ee_pose.ranges.pos_x = (0.15, 0.35)
        self.commands.ee_pose.ranges.pos_y = (-0.2, 0.2)
        self.commands.ee_pose.ranges.pos_z = (0.15, 0.5)
        self.commands.ee_pose.ranges.pitch = (0.0, 0.0)


@configclass
class ArctosReachEnvCfg_PLAY(ArctosReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False