#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Franka 로봇을 사용한 자동 시뮬레이션 Pick & Place
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
from isaacsim.robot.manipulators.examples.franka.tasks import PickPlace


def main():
    """Franka 로봇을 사용한 자동 시뮬레이션 Pick & Place"""
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
    
    print("Franka 시스템 설정 완료. 자동 시뮬레이션 시작...")
    
    # 간단한 실행 루프
    step_count = 0
    max_steps = 1000
    
    while simulation_app.is_running() and step_count < max_steps:
        world.step(render=True)
        
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
    
    print("Franka 자동 시뮬레이션 완료")
    simulation_app.close()


if __name__ == "__main__":
    main() 