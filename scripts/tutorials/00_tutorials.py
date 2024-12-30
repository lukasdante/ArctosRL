# Launch Isaac Sim
import argparse
from omni.isaac.lab.app import AppLauncher

# Create an argument parser for launching app
parser = argparse.ArgumentParser(description='RL scene configuration for Arctos robot arm.')
# Add custom arguments
parser.add_argument('--cone_height', type=float, default=0.5, help='Height of the cone.')
parser.add_argument('--width', type=int, default=1280, help='Width of the viewport.')
parser.add_argument('--height', type=int, default=720, help='Height of the viewport.')
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import Isaac Lab modules
import torch
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.sim import SimulationCfg, SimulationContext
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # Lights
    cfg_light_distant = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # Spawn a Group of Rigid Objects
    prim_utils.create_prim('/World/Objects', 'Xform')
    # spawn a red cone
    cfg_cone = sim_utils.ConeCfg(radius=0.15, height=args_cli.cone_height, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)))
    cfg_cone.func('/World/Objects/Cone1', cfg_cone, translation=(-1.0, 1.0, 1.0))
    cfg_cone.func('/World/Objects/Cone2', cfg_cone, translation=(-1.0, -1.0, 1.0))
    # spawn a green cone with colliders and rigid body
    cfg_cone_rigid = sim_utils.ConeCfg(radius=0.15,
                                       height=0.5,
                                       rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                       mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                                       collision_props=sim_utils.CollisionPropertiesCfg(),
                                       visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                                       )
    cfg_cone_rigid.func('/World/Objects/ConeRigid', cfg_cone_rigid, translation=(-0.2, 0.0, 2.0), orientation=(0.5, 0.0, 0.5, 0.0))
    # spawn a blue cuboid with deformable body
    cfg_cuboid_deformable = sim_utils.MeshCuboidCfg(size=(0.2, 0.5, 0.2),
                                                    deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
                                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                                                    physics_material=sim_utils.DeformableBodyMaterialCfg()
                                                    )
    cfg_cuboid_deformable.func('/World/Objects/CuboidDeformable', cfg_cuboid_deformable, translation=(0.15, 0.0, 2.0))
    # spawn a USD file
    cfg = sim_utils.UsdFileCfg(usd_path=f'{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd')
    cfg.func('/World/Objects/Table', cfg, translation=(0.0, 0.0, 1.05))

    # Spawn a Group of Rigid Objects with the same physics
    origins = [[0.25, 0.25, 2.0], [-0.25, 0.25, 2.0], [-0.25, -0.25, 2.0], [0.25, -0.25, 2.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f'/World/Origin{i}', 'Xform', translation=origin)

    cone_cfg = RigidObjectCfg(prim_path='/World/Origin.*/Cone',
                              spawn=sim_utils.ConeCfg(
                                  radius=0.1,
                                  height=0.2,
                                  rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                  mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                                  collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                              ),
                              init_state=RigidObjectCfg.InitialStateCfg(),
                              )
    cone_object = RigidObject(cfg=cone_cfg)

    scene_entities = {"cone": cone_object}
    return scene_entities, origins

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], origins: torch.Tensor):
    # Extract objects
    cone_object = entities['cone']

    # 
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        if count % 250 == 0:
            # reset counters
            sim_time = 0
            count = 0
            
            # reset root state
            root_state = cone_object.data.default_root_state.clone()

            # sample a random position on a cylinder around origins
            root_state[:, :3] += origins
            root_state[:, :3] += math_utils.sample_cylinder(radius=0.1,
                                                            h_range=(0.25, 0.5),
                                                            size=cone_object.num_instances,
                                                            device=cone_object.device)
            # write root state to simulation
            cone_object.write_root_state_to_sim(root_state)
            # reset buffers
            cone_object.reset()
            print('[INFO]: Resetting object state...')
        # apply sim data
        cone_object.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        # update buffers
        cone_object.update(sim_dt)
        # print root position
        if count % 50 == 0:
            print(f'Root position (in world): {cone_object.data.root_state_w[:, :3]}')

def main():

    # Initialize simulation context w/ timestep = 0.01
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set the main camera on load
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

    # Design the scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    # Play the simulator
    sim.reset()

    run_simulator(sim, scene_entities, scene_origins)

if __name__ == '__main__':
    main()
    simulation_app.close()