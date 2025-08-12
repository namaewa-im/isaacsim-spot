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
import json
import pandas as pd
from datetime import datetime
import time

class ManualSpotDataCollector:
    def __init__(self):
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 500.0,
            rendering_dt=10.0 / 500.0
        )
        
        # 출력 디렉토리 설정 (날짜별로 구분)
        current_date = datetime.now().strftime("%Y%m%d")
        self.output_dir = os.path.expanduser(f"~/isaacsim/spot_lerobot_data/{current_date}")
        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # 데이터 수집 설정
        self.data_collection_interval = 10  # 매 10 스텝마다 수집 (0.02초)
        self.step_count = 0
        self.global_index = 0
        
        # 데이터 수집 상태
        self.is_collecting = False
        self.current_task = None
        self.episode_data = {
            'observations': [],
            'actions': [],
            'episode_index': 0,
            'frame_index': 0,
            'timestamps': [],
            'next_done': [],
            'index': [],
            'task_index': 0,
            'task_name': '',
            'start_time': None
        }
        
        # 태스크 정의 (자유롭게 수정 가능)
        self.tasks = [
            {"name": "forward_walk", "task_index": 0},
            {"name": "backward_walk", "task_index": 1},
            {"name": "left_walk", "task_index": 2},
            {"name": "right_walk", "task_index": 3},
            {"name": "rotate_left", "task_index": 4},
            {"name": "rotate_right", "task_index": 5},
            {"name": "custom_task", "task_index": 6},
        ]
        
        # 현재 태스크 인덱스
        self.current_task_index = 0
        
        # 키보드 입력 처리
        self.input_keyboard_mapping = {
            "NUMPAD_8": np.array([2.0, 0.0, 0.0]),      # 전진
            "UP": np.array([2.0, 0.0, 0.0]),
            "NUMPAD_2": np.array([-2.0, 0.0, 0.0]),     # 후진
            "DOWN": np.array([-2.0, 0.0, 0.0]),
            "NUMPAD_4": np.array([0.0, 2.0, 0.0]),      # 좌측
            "LEFT": np.array([0.0, 2.0, 0.0]),
            "NUMPAD_6": np.array([0.0, -2.0, 0.0]),     # 우측
            "RIGHT": np.array([0.0, -2.0, 0.0]),
            "NUMPAD_7": np.array([0.0, 0.0, 2.0]),      # 좌회전
            "NUMPAD_9": np.array([0.0, 0.0, -2.0]),     # 우회전
            "SPACE": np.array([0.0, 0.0, 0.0]),         # 정지
        }
        
        # 특수 키 매핑
        self.special_keys = {
            "ENTER": "start_stop_collection",    # 데이터 수집 시작/정지
            "TAB": "next_task",                  # 다음 태스크로 변경
            "SHIFT_TAB": "prev_task",            # 이전 태스크로 변경
            "S": "save_episode",                 # 현재 에피소드 저장
            "R": "reset_robot",                  # 로봇 위치 리셋
            "Q": "quit",                         # 종료
        }
        
        # 씬 설정
        self.setup_scene()
        
        # 키보드 콜백 등록
        self.setup_keyboard_callback()
        
        print("=== 수동 Spot 데이터 수집기 ===")
        print("키보드 조작법:")
        print("  방향키/NUMPAD: 로봇 이동")
        print("  ENTER: 데이터 수집 시작/정지")
        print("  TAB: 다음 태스크")
        print("  S: 현재 에피소드 저장")
        print("  R: 로봇 위치 리셋")
        print("  Q: 종료")
        print(f"현재 태스크: {self.tasks[self.current_task_index]['name']}")
        
    def setup_scene(self):
        """씬 설정"""
        # 기본 지형 추가
        self.world.scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.2,
            dynamic_friction=0.2,
            restitution=0.01,
        )
        
        # DomeLight 추가
        self.setup_dome_light()
        
        # Spot 로봇 생성
        self.spot = SpotFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            position=np.array([0, 0, 0.8]),
        )
        
        # 카메라 설정
        self.setup_camera_on_body()
        
    def setup_dome_light(self):
        """DomeLight 설정"""
        import omni.usd
        from pxr import Sdf
        
        stage = omni.usd.get_context().get_stage()
        dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
        dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(450.0)
        print("DomeLight가 성공적으로 추가되었습니다.")
        
    def setup_camera_on_body(self):
        """로봇 본체에 카메라 설정"""
        self.spot_camera = Camera(
            prim_path="/World/Spot/body/camera",
            frequency=60,
            resolution=(256, 256),
            position=np.array([0.3, 0, 0.2]),
            orientation=euler_angles_to_quats(np.array([0, 0, 0])),
        )
        
    def setup_keyboard_callback(self):
        """키보드 콜백 설정"""
        self.input = carb.input.acquire_input_interface()
        self.keyboard = omni.appwindow.acquire_input_interface()
        
        # 키보드 콜백 등록
        self.keyboard.subscribe_to_keyboard_events(self._on_keyboard_event)
        
    def _on_keyboard_event(self, event):
        """키보드 이벤트 처리"""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # 특수 키 처리
            if event.input in self.special_keys:
                self._handle_special_key(self.special_keys[event.input])
                return
                
            # 일반 이동 키 처리
            if event.input in self.input_keyboard_mapping:
                command = self.input_keyboard_mapping[event.input]
                self.spot.base_command = command
                print(f"명령 실행: {event.input} -> {command}")
                
    def _handle_special_key(self, action):
        """특수 키 처리"""
        if action == "start_stop_collection":
            self.toggle_data_collection()
        elif action == "next_task":
            self.next_task()
        elif action == "prev_task":
            self.prev_task()
        elif action == "save_episode":
            self.save_current_episode()
        elif action == "reset_robot":
            self.reset_robot_position()
        elif action == "quit":
            self.quit_simulation()
            
    def toggle_data_collection(self):
        """데이터 수집 시작/정지"""
        if not self.is_collecting:
            self.start_data_collection()
        else:
            self.stop_data_collection()
            
    def start_data_collection(self):
        """데이터 수집 시작"""
        self.is_collecting = True
        self.episode_data = {
            'observations': [],
            'actions': [],
            'episode_index': 0,
            'frame_index': 0,
            'timestamps': [],
            'next_done': [],
            'index': [],
            'task_index': self.current_task_index,
            'task_name': self.tasks[self.current_task_index]['name'],
            'start_time': time.time()
        }
        print(f"데이터 수집 시작: {self.tasks[self.current_task_index]['name']}")
        
    def stop_data_collection(self):
        """데이터 수집 정지"""
        self.is_collecting = False
        print("데이터 수집 정지")
        
    def next_task(self):
        """다음 태스크로 변경"""
        self.current_task_index = (self.current_task_index + 1) % len(self.tasks)
        print(f"태스크 변경: {self.tasks[self.current_task_index]['name']}")
        
    def prev_task(self):
        """이전 태스크로 변경"""
        self.current_task_index = (self.current_task_index - 1) % len(self.tasks)
        print(f"태스크 변경: {self.tasks[self.current_task_index]['name']}")
        
    def save_current_episode(self):
        """현재 에피소드 저장"""
        if len(self.episode_data['observations']) > 0:
            self.save_episode_data()
            print(f"에피소드 저장 완료: {len(self.episode_data['observations'])} 프레임")
        else:
            print("저장할 데이터가 없습니다.")
            
    def reset_robot_position(self):
        """로봇 위치 리셋"""
        self.spot.robot.set_world_pose(position=np.array([0, 0, 0.8]), orientation=np.array([1, 0, 0, 0]))
        print("로봇 위치 리셋")
        
    def quit_simulation(self):
        """시뮬레이션 종료"""
        print("시뮬레이션 종료")
        simulation_app.close()
        
    def collect_robot_state(self):
        """로봇 상태 수집 (Spot의 _compute_observation과 동일한 48차원 구조)"""
        # 기본 물리 상태
        lin_vel_I = self.spot.robot.get_linear_velocity()
        ang_vel_I = self.spot.robot.get_angular_velocity()
        pos_IB, q_IB = self.spot.robot.get_world_pose()
        
        # 관절 상태
        joint_pos = self.spot.robot.get_joint_positions()
        joint_vel = self.spot.robot.get_joint_velocities()
        
        # 현재 명령
        command = self.spot.base_command
        
        # 로봇 본체 좌표계로 변환
        from isaacsim.core.utils.rotations import quat_to_rot_matrix
        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)  # 로봇 좌표계 선형 속도
        ang_vel_b = np.matmul(R_BI, ang_vel_I)  # 로봇 좌표계 각속도
        
        # 중력 방향 (로봇 좌표계)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))
        
        # 기본 관절 위치 (Spot의 default_pos)
        default_pos = np.array([0.0, 0.8, -1.6, 0.0, 0.8, -1.6, 0.0, 0.8, -1.6, 0.0, 0.8, -1.6])
        
        # 이전 액션 (첫 번째 프레임에서는 0으로 초기화)
        if not hasattr(self, 'previous_action'):
            self.previous_action = np.zeros(12)
        
        # Spot의 _compute_observation과 동일한 48차원 관찰 벡터 구성
        obs = np.zeros(48)
        obs[:3] = lin_vel_b                    # 기본 선형 속도 (3)
        obs[3:6] = ang_vel_b                   # 기본 각속도 (3)
        obs[6:9] = gravity_b                   # 중력 방향 (3)
        obs[9:12] = command                    # 명령 (v_x, v_y, w_z) (3)
        obs[12:24] = joint_pos - default_pos   # 관절 위치 오프셋 (12)
        obs[24:36] = joint_vel                 # 관절 속도 (12)
        obs[36:48] = self.previous_action      # 이전 액션 (12)
        
        return obs
        
    def collect_action_data(self):
        """액션 데이터 수집"""
        # Spot의 내부 액션 (12차원 관절 위치 명령)
        if hasattr(self.spot, 'action') and self.spot.action is not None:
            return self.spot.action
        else:
            # 액션이 없으면 0으로 채움
            return np.zeros(12)
            
    def capture_camera_image(self):
        """카메라 이미지 캡처 및 저장"""
        if self.spot_camera is not None:
            try:
                rgb_data = self.spot_camera.get_rgba()
                if rgb_data is not None and hasattr(rgb_data, 'shape') and len(rgb_data.shape) == 3:
                    # RGBA에서 RGB로 변환
                    rgb_image = rgb_data[:, :, :3]
                    
                    # 이미지 저장
                    image_filename = f"task{self.current_task_index:02d}_ep{self.episode_data['episode_index']:04d}_frame{self.episode_data['frame_index']:03d}.png"
                    filepath = os.path.join(self.images_dir, image_filename)
                    
                    plt.imsave(filepath, rgb_image)
                    return image_filename
                    
            except Exception as e:
                print(f"카메라 이미지 캡처 오류: {e}")
                
        return None
        
    def save_episode_data(self):
        """에피소드 데이터 저장"""
        if len(self.episode_data['observations']) == 0:
            return
            
        # 데이터프레임 생성
        df = pd.DataFrame({
            'observation': self.episode_data['observations'],
            'action': self.episode_data['actions'],
            'episode_index': self.episode_data['episode_index'],
            'frame_index': self.episode_data['frame_index'],
            'timestamp': self.episode_data['timestamps'],
            'next.done': self.episode_data['next_done'],
            'index': self.episode_data['index'],
            'task_index': self.episode_data['task_index']
        })
        
        # 파일명 생성
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"task{self.current_task_index:02d}_ep{self.episode_data['episode_index']:04d}_{timestamp}.parquet"
        filepath = os.path.join(self.output_dir, filename)
        
        # 저장
        df.to_parquet(filepath, index=False)
        
        # 메타데이터 저장
        metadata = {
            'task_name': self.episode_data['task_name'],
            'task_index': self.episode_data['task_index'],
            'episode_index': self.episode_data['episode_index'],
            'frame_count': len(self.episode_data['observations']),
            'duration': time.time() - self.episode_data['start_time'],
            'timestamp': timestamp
        }
        
        metadata_filename = f"task{self.current_task_index:02d}_ep{self.episode_data['episode_index']:04d}_{timestamp}_meta.json"
        metadata_filepath = os.path.join(self.output_dir, metadata_filename)
        
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # 에피소드 인덱스 증가
        self.episode_data['episode_index'] += 1
        
    def on_physics_step(self, step_size):
        """물리 스텝 콜백"""
        # 로봇 제어
        self.spot.forward(step_size, self.spot.base_command)
        
        # 데이터 수집 (매 10스텝마다)
        if self.is_collecting and self.step_count % self.data_collection_interval == 0:
            # 상태 수집
            state = self.collect_robot_state()
            action = self.collect_action_data()
            image_filename = self.capture_camera_image()
            
            # 타임스탬프 계산
            timestamp = (time.time() - self.episode_data['start_time']) * 1000  # 밀리초
            
            # 데이터 저장
            self.episode_data['observations'].append(state.tolist())
            self.episode_data['actions'].append(action.tolist())
            self.episode_data['timestamps'].append(timestamp)
            self.episode_data['next_done'].append(False)  # 마지막에만 True로 설정
            self.episode_data['index'].append(self.global_index)
            
            # 이전 액션 업데이트 (다음 프레임에서 사용)
            self.previous_action = action.copy()
            
            # 프레임 인덱스 증가
            self.episode_data['frame_index'] += 1
            self.global_index += 1
            
            # 진행 상황 출력 (매 50프레임마다)
            if self.episode_data['frame_index'] % 50 == 0:
                print(f"수집 중: {self.episode_data['frame_index']} 프레임")
                
        self.step_count += 1
        
    def run(self):
        """메인 실행 루프"""
        self.world.reset()
        
        while simulation_app.is_running():
            self.world.step(render=True)
            
            # ESC 키로 종료
            if carb.input.is_keyboard_event_pressed(carb.input.KeyboardInput.ESCAPE):
                break
                
        simulation_app.close()

if __name__ == "__main__":
    collector = ManualSpotDataCollector()
    collector.run()
