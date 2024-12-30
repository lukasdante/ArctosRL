import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg

ARCTOS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path='no_gripper.usd',
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
        visual_material= sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.75)),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "a": 0.0,
            "b": 3.14,
            "c": 0.0,
        },
        joint_vel={
            '.*': 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["x", "y", "z", "a", "b", "c"],
            stiffness=None,
            damping=None
        )
    },
    soft_joint_pos_limit_factor=1.05
)