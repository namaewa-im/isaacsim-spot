#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
import omni.appwindow
import omni.usd
from isaacsim.core.api import World
import isaacsim.core.utils.prims as prim_utils
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
import matplotlib.pyplot as plt
import os

class SpotCameraExample:
    def __init__(self):
        self.world = None
        self.spot = None
        self.spot_camera = None
        self.base_command = np.array([0.0, 0.0, 0.0])
        self.physics_ready = False
        self.first_step = True
        self.reset_needed = False
        
        # 키보드 매핑 설정 (5.spot_teleop_standalone.py와 동일)
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
        
        # 캡처를 위한 설정
        self.capture_interval = 100  # 매 100 스텝마다 캡처
        self.step_count = 0
        self.capture_count = 0
        self.output_dir = os.path.expanduser("~/isaacsim/spot_camera_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
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
        
        # Spot 로봇 생성
        self.spot = SpotFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            position=np.array([0, 0, 0.8]),
        )
        
        # 카메라 설정도 여기서 한 번만 실행
        self.setup_camera_on_body()
        
    def setup_camera_on_body(self):
        """Spot의 body에 카메라를 부착하는 함수 (Xform 프림 방식)"""
        print("Spot 로봇의 body에 카메라 부착 중 (Xform 프림 방식)...")
        
        # Spot body에 상대적인 카메라 위치와 방향
        camera_name = "/World/Spot/body/camera_rig"
        
        # Xform 프림 생성 (카메라 리그)
        camera_prim = prim_utils.create_prim(
            camera_name,
            "Xform",
            translation=np.array([0.3, 0.0, 0.2]),  # body 앞쪽 위
            orientation=euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
        )
        
        print(f"카메라 Xform 프림 생성됨: {camera_name}")
        
        # Xform 프림 아래에 실제 카메라 생성
        self.spot_camera = Camera(
            prim_path=f"{camera_name}/camera",
            frequency=25,
            resolution=(640, 480),
        )
        
        print(f"카메라가 {camera_name}/camera 경로에 생성되었습니다.")
            
    def on_physics_step(self, step_size):
        """물리 스텝 콜백"""
        if self.first_step:
            self.spot.initialize()
            self.spot_camera.initialize()
            self.first_step = False
            print("Spot과 카메라 초기화 완료")
            
        elif self.reset_needed:
            self.world.reset(True)
            self.reset_needed = False
            self.first_step = True
            # 리셋 후 명령 초기화
            self.base_command = np.array([0.0, 0.0, 0.0])
            print("시뮬레이션 리셋됨 - 명령 초기화")
            
        else:
            # Spot 로봇 제어
            self.spot.forward(step_size, self.base_command)
            
            # 주기적으로 카메라 이미지 캡처
            self.step_count += 1
            if self.step_count % self.capture_interval == 0:
                self.capture_camera_image()
                
    def capture_camera_image(self):
        """카메라 이미지 캡처 및 저장"""
        if self.spot_camera is not None:
            try:
                # RGB 이미지 가져오기
                rgb_data = self.spot_camera.get_rgba()
                if rgb_data is not None:
                    # RGBA에서 RGB로 변환 (알파 채널 제거)
                    rgb_image = rgb_data[:, :, :3]
                    
                    # 이미지 저장
                    filename = f"spot_camera_{self.capture_count:04d}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    
                    plt.imsave(filepath, rgb_image)
                    print(f"카메라 이미지 저장됨: {filepath}")
                    
                    self.capture_count += 1
                    
            except Exception as e:
                print(f"카메라 이미지 캡처 중 오류 발생: {e}")
                
    def sub_keyboard_event(self, event, *args, **kwargs):
        """키보드 이벤트 처리"""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # 키를 누를 때 명령 증가
            if event.input.name in self.input_keyboard_mapping:
                self.base_command += self.input_keyboard_mapping[event.input.name]
                print(f"키 {event.input.name} 누름 - 명령: {self.base_command}")
                
            # 'C' 키를 누르면 수동으로 카메라 이미지 캡처
            elif event.input.name == "C":
                print("수동 카메라 캡처 요청")
                self.capture_camera_image()
                
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # 키를 놓을 때 명령 감소
            if event.input.name in self.input_keyboard_mapping:
                self.base_command -= self.input_keyboard_mapping[event.input.name]
                print(f"키 {event.input.name} 놓음 - 명령: {self.base_command}")
        return True
        
    def main(self):
        """메인 실행 함수"""
        print("Spot 로봇 body 카메라 부착 예제 시작 (Xform 프림 방식)")
        print("키보드 컨트롤:")
        print("  방향키 또는 NUMPAD 8/2/4/6: 전진/후진/좌회전/우회전")
        print("  N/M 또는 NUMPAD 7/9: 좌회전/우회전")
        print("  C: 수동 카메라 캡처")
        print("  ESC: 종료")
        print(f"자동 캡처 간격: {self.capture_interval} 스텝")
        print(f"이미지 저장 경로: {self.output_dir}")
        
        # World 설정
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 500.0,
            rendering_dt=10.0 / 500.0
        )
        self.world.reset()
        
        # 씬 설정 (카메라 포함)
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
            if step_count % 1000 == 0:
                print(f"진행 중... {step_count} 스텝, 현재 명령: {self.base_command}, 캡처된 이미지: {self.capture_count}")
                
        print(f"Spot 로봇 카메라 예제 완료 - 총 {self.capture_count}개 이미지 캡처됨")
        print(f"저장된 이미지들: {self.output_dir}")
        simulation_app.close()

if __name__ == "__main__":
    spot_camera_example = SpotCameraExample()
    spot_camera_example.main()