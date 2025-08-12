#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
import omni.appwindow  # Contains handle to keyboard
from isaacsim.core.api import World
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy

class QuadrupedKeyboardStandalone:
    def __init__(self):
        self.world = None
        self.spot = None
        self.base_command = np.array([0.0, 0.0, 0.0])
        self.physics_ready = False
        self.first_step = True
        self.reset_needed = False
        
        # 키보드 매핑 설정 (interactive 예제와 동일)
        self.input_keyboard_mapping = {
            # forward command
            "NUMPAD_8": np.array([2.0, 0.0, 0.0]),
            "UP": np.array([2.0, 0.0, 0.0]),
            # back command
            "NUMPAD_2": np.array([-2.0, 0.0, 0.0]),
            "DOWN": np.array([-2.0, 0.0, 0.0]),
            # left command
            "NUMPAD_6": np.array([0.0, -2.0, 0.0]),
            "RIGHT": np.array([0.0, -2.0, 0.0]),
            # right command
            "NUMPAD_4": np.array([0.0, 2.0, 0.0]),
            "LEFT": np.array([0.0, 2.0, 0.0]),
            # yaw command (positive)
            "NUMPAD_7": np.array([0.0, 0.0, 2.0]),
            "N": np.array([0.0, 0.0, 2.0]),
            # yaw command (negative)
            "NUMPAD_9": np.array([0.0, 0.0, -2.0]),
            "M": np.array([0.0, 0.0, -2.0]),
        }
        
        # 키보드 입력 설정
        self.appwindow = omni.appwindow.get_default_app_window()
        self.input = carb.input.acquire_input_interface()
        self.keyboard = self.appwindow.get_keyboard()
        self.sub_keyboard = self.input.subscribe_to_keyboard_events(
            self.keyboard, self.sub_keyboard_event
        )
        
    def setup_scene(self):
        """씬 설정"""
        self.world.scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.2,
            dynamic_friction=0.2,
            restitution=0.01,
        )
        
        self.spot = SpotFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            position=np.array([0, 0, 0.8]),
        )
        
    def on_physics_step(self, step_size):
        """물리 스텝 콜백"""
        if self.first_step:
            self.spot.initialize()
            self.first_step = False
        elif self.reset_needed:
            self.world.reset(True)
            self.reset_needed = False
            self.first_step = True
            # 리셋 후 명령 초기화
            self.base_command = np.array([0.0, 0.0, 0.0])
            print("시뮬레이션 리셋됨 - 명령 초기화")
        else:
            self.spot.forward(step_size, self.base_command)
            
    def sub_keyboard_event(self, event, *args, **kwargs):
        """키보드 이벤트 처리"""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # 키를 누를 때 명령 증가
            if event.input.name in self.input_keyboard_mapping:
                self.base_command += self.input_keyboard_mapping[event.input.name]
                print(f"키 {event.input.name} 누름 - 명령: {self.base_command}")
                
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # 키를 놓을 때 명령 감소
            if event.input.name in self.input_keyboard_mapping:
                self.base_command -= self.input_keyboard_mapping[event.input.name]
                print(f"키 {event.input.name} 놓음 - 명령: {self.base_command}")
        return True
        
    def main(self):
        """메인 실행 함수"""
        print("Spot 로봇 키보드 제어 Standalone 시작")
        print("키보드 컨트롤:")
        print("  방향키 또는 NUMPAD 8/2/4/6: 전진/후진/좌회전/우회전")
        print("  N/M 또는 NUMPAD 7/9: 좌회전/우회전")
        print("  ESC: 종료")
        
        # World 설정
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 500.0,
            rendering_dt=10.0 / 500.0
        )
        self.world.reset()
        
        # 씬 설정
        self.setup_scene()
        self.world.reset()
        
        # 물리 콜백 추가
        self.world.add_physics_callback("physics_step", callback_fn=self.on_physics_step)
        
        # 시뮬레이션 시작
        self.world.play()
        
        # 메인 루프
        step_count = 0
        max_steps = 100000  # 무한 루프 방지
        
        while simulation_app.is_running() and step_count < max_steps:
            self.world.step(render=True)
            step_count += 1
            
            # 시뮬레이션이 중지되면 리셋 플래그 설정
            if self.world.is_stopped():
                self.reset_needed = True
                
            # 주기적으로 상태 출력
            if step_count % 500 == 0:
                print(f"진행 중... {step_count} 스텝, 현재 명령: {self.base_command}")
                
        print("Spot 로봇 키보드 제어 완료")
        simulation_app.close()

if __name__ == "__main__":
    quadruped = QuadrupedKeyboardStandalone()
    quadruped.main()