# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import matplotlib.pyplot as plt
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.storage.native import get_assets_root_path
import omni.usd
import omni.kit.commands
import omni.replicator.core as rep
import os

# 월드 생성
my_world = World(stage_units_in_meters=1.0)

# 에셋 경로 설정
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    print("Could not find Isaac Sim assets folder")
    simulation_app.close()
    exit()

# Franka 로봇 로드
asset_path = assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka")

# 그리퍼 설정
gripper = ParallelGripper(
    end_effector_prim_path="/World/Franka/panda_rightfinger",
    joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    joint_opened_positions=np.array([0.05, 0.05]),
    joint_closed_positions=np.array([0.02, 0.02]),
    action_deltas=np.array([0.01, 0.01]),
)

# 로봇 생성
my_franka = my_world.scene.add(
    SingleManipulator(
        prim_path="/World/Franka", 
        name="my_franka", 
        end_effector_prim_name="panda_rightfinger", 
        gripper=gripper
    )
)

# 테스트용 큐브 생성
cube = my_world.scene.add(
    DynamicCuboid(
        name="cube",
        position=np.array([0.3, 0.3, 0.3]),
        prim_path="/World/Cube",
        scale=np.array([0.0515, 0.0515, 0.0515]),
        size=1.0,
        color=np.array([0, 0, 255]),
    )
)

# DomeLight 추가하여 전체 환경을 밝게 만들기
def create_or_get_dome_light(scene_path="/World/DomeLight", intensity=1000):
    def create_dome_light():
        omni.kit.commands.execute(
            "CreatePrim",
            prim_path=scene_path,
            prim_type="DomeLight",
            attributes={"inputs:intensity": intensity, "inputs:texture:format": "latlong"},
        )
    
    # DomeLight 생성
    create_dome_light()

# DomeLight 생성
create_or_get_dome_light("/World/DomeLight", intensity=1000)

# 월드 초기화
my_world.scene.add_default_ground_plane()
my_world.reset()

# 로봇 초기화 후 엔드 이펙터 위치 가져오기
my_franka.initialize()

# 엔드 이펙터에 부착된 카메라
end_effector_pose = my_franka.end_effector.get_world_pose()
camera_position = end_effector_pose[0] + np.array([0, 0, 0.1])  # 엔드 이펙터 위 10cm

end_effector_camera = Camera(
    prim_path="/World/Franka/end_effector_camera",
    position=camera_position,
    frequency=20,
    resolution=(640, 480),
    orientation=euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
)

# 카메라 초기화
end_effector_camera.initialize()

# Replicator를 사용한 영상 저장 설정
output_dir = os.path.expanduser("~/isaacsim/video")
os.makedirs(output_dir, exist_ok=True)

# Replicator writer 설정
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir=output_dir,
    rgb=True,
    bounding_box_2d_tight=False,
    semantic_segmentation=False,
    distance_to_image_plane=False,
    camera_params=False,
    image_output_format="png"
)

# Render product 생성 및 writer에 연결
render_product = rep.create.render_product("/World/Franka/end_effector_camera", (640, 480))
writer.attach([render_product])

# 시뮬레이션 루프
i = 0
reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)
    
    # 카메라 이미지 가져오기
    if i % 30 == 0:  # 30프레임마다 한 번씩
        print(f"Frame {i}")
        
        # 엔드 이펙터 카메라 이미지
        end_effector_rgba = end_effector_camera.get_rgba()
        print(f"End effector camera image shape: {end_effector_rgba.shape}")
        
        # 이미지 시각화 (선택사항)
        if i == 60:  # 60프레임에서 이미지 저장
            plt.figure(figsize=(10, 10))
            plt.imshow(end_effector_rgba[:, :, :3])
            plt.title("End Effector Camera")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig("end_effector_camera.png", dpi=300)
            plt.close()
            print("Camera image saved to end_effector_camera.png")
    
    # 리셋 처리
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_needed = False
    
    i += 1
    
    # 테스트 모드에서는 300프레임 후 종료
    if i > 300:
        break

# Replicator writer 정리
writer.detach()
simulation_app.close() 