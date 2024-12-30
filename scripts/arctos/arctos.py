import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg

# Create the Articulation configuration for Arctos robot arm using the USD file

ARCTOS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # An error shows up when the base Arctos USD file is used in the gripper
        # so I removed the gripper since it isn't a required rigid body in reach task
        # This USD file is originally from hidara, but I factored down the max joint velocity
        # of each joint to only 114.5 deg/s or 2 rad/s. You can set it arbitrarily
        # on your own with Isaac Sim > Physics > Advanced > Maximum Joint Velocity (deg/s)
        usd_path='no_gripper.usd',

        # No contact sensors were used
        activate_contact_sensors=False,

        # Set the rigid properties of the robot arm
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),

        # Set articulation properties of the robot arm
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),

        # Set visual material, original USD is green, I just changed it to white
        visual_material= sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.75)),
    ),

    # Set initial state to 0 for all joints except joint position of b which is 180 degrees
    # You can see this in the USD file by enabling Physics Authoring Toolbar
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

    # Create an actuator model, most straightforward is ImplicitActuatorCfg
    # Set stiffness and damping to both None, if None, it uses the stiffness and damping
    # from the USD file which is gain tuned with stiff gains
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["x", "y", "z", "a", "b", "c"],
            stiffness=None,
            damping=None
        )
    },

    # Adjusts the joint position limits by a small factor to avoid going beyond the limits
    soft_joint_pos_limit_factor=1.05
)