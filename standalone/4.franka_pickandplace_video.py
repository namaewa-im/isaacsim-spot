#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Franka 로봇을 사용한 자동 시뮬레이션 Pick & Place (End Effector 카메라 포함)
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import os
from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
from isaacsim.robot.manipulators.examples.franka.tasks import PickPlace
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
import omni.replicator.core as rep
from datetime import datetime

def main():
    """Franka 로봇을 사용한 자동 시뮬레이션 Pick & Place (End Effector 카메라 포함)"""
    print("Franka 로봇 자동 시뮬레이션 Pick & Place 시작")
    
    # World 설정 (50Hz 렌더링)
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0/60.0,    # 60Hz 물리 시뮬레이션
        rendering_dt=1.0/50.0    # 50Hz 렌더링
    )
    task = PickPlace()
    world.add_task(task)
    world.reset()
    world.play()  # 자동 시작
    
    # 로봇과 컨트롤러 설정
    task_params = task.get_params()
    robot = world.scene.get_object(task_params["robot_name"]["value"])
    controller = PickPlaceController(
        name="pick_place_controller",
        gripper=robot.gripper,
        robot_articulation=robot
    )
    articulation_controller = robot.get_articulation_controller()
    
    # End Effector 카메라 설정 - 초기 위치
    end_effector_pose = robot.end_effector.get_world_pose()
    camera_position = end_effector_pose[0] + np.array([0, 0, 0.1])  # 엔드 이펙터 위 10cm
    
    end_effector_camera = Camera(
        prim_path="/World/Franka/end_effector_camera",
        position=camera_position,
        frequency=50,  # 50Hz로 설정
        resolution=(640, 480),
        orientation=euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
    )
    
    # 카메라 초기화
    end_effector_camera.initialize()
    
    # 이미지 저장 디렉토리 설정
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.expanduser(f"~/isaacsim/video/franka/pickandplace/{time}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"이미지 저장 경로: {output_dir}")
    
    # Replicator를 사용한 영상 저장 설정
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
    
    print("Franka 시스템 설정 완료. 자동 시뮬레이션 시작...")
    
    # 간단한 실행 루프
    step_count = 0
    max_steps = 1000
    
    while simulation_app.is_running() and step_count < max_steps:
        world.step(render=True)
        
        # 카메라를 end effector에 동적으로 붙이기
        current_end_effector_pose = robot.end_effector.get_world_pose()
        current_camera_position = current_end_effector_pose[0] + np.array([0, 0, 0.1])
        
        # 카메라 위치 업데이트 (end effector 따라가기)
        end_effector_camera.set_world_pose(
            position=current_camera_position,
            orientation=current_end_effector_pose[1]  # end effector의 회전을 따라감
        )
        
        # 카메라 이미지 가져오기 및 저장
        if step_count % 30 == 0:  # 30프레임마다 한 번씩
            print(f"Frame {step_count}")
            print(f"End effector position: {current_end_effector_pose[0]}")
            print(f"Camera position: {current_camera_position}")
            
            # 엔드 이펙터 카메라 이미지
            end_effector_rgba = end_effector_camera.get_rgba()
            if end_effector_rgba is not None:
                print(f"End effector camera image shape: {end_effector_rgba.shape}")
        
        observations = world.get_observations()
        actions = controller.forward(
            picking_position=observations[task_params["cube_name"]["value"]]["position"],
            placing_position=observations[task_params["cube_name"]["value"]]["target_position"],
            current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
            end_effector_offset=np.array([0, 0.005, 0])  # Franka용 오프셋 조정
        )
        
        articulation_controller.apply_action(actions)
        
        if controller.is_done():
            print(f"Franka Pick & Place 완료! (총 {step_count} 스텝)")
            break
        
        step_count += 1
        
        if step_count % 100 == 0:
            print(f"진행 중... {step_count} 스텝")
    
    # Replicator writer 정리
    writer.detach()
    print("Franka 자동 시뮬레이션 완료")
    print(f"총 {step_count} 스텝의 이미지가 {output_dir}에 저장되었습니다.")
    simulation_app.close()


if __name__ == "__main__":
    main() 