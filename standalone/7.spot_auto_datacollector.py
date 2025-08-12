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
import random

class LeRobotSpotDataCollector:
    def __init__(self):
        self.world = None
        self.spot = None
        self.spot_camera = None
        self.base_command = np.array([0.0, 0.0, 0.0])
        self.physics_ready = False
        self.first_step = True
        
        # 데이터 수집 설정
        self.data_collection_interval = 10  # 매 10 스텝마다 수집 (0.02초)
        self.episode_length = 250  # 에피소드 길이 (5초 = 250 * 0.02초)
        self.episodes_per_task = 5  # 각 태스크당 5개 에피소드로 변경
        self.step_count = 0
        self.episode_count = 0
        self.frame_count = 0
        self.global_index = 0
        
        # 카메라 워밍업을 위한 변수 추가
        self.camera_warmup_steps = 50  # 카메라 워밍업을 위한 스텝 수
        self.camera_ready = False
        
        # LeRobot 데이터 구조 (Spot에 맞게 수정)
        self.episode_data = {
            'observation.state': [],      # 48차원 상태 벡터
            'observation.images': [],  # 이미지 파일명
            'action': [],                 # 12차원 액션 벡터
            'episode_index': [],
            'frame_index': [],
            'timestamp': [],
            'next.done': [],
            'index': [],
            'task_index': []
        }
        
        # 출력 디렉토리 설정 (날짜별로 구분)
        current_date = datetime.now().strftime("%Y%m%d")
        self.output_dir = os.path.expanduser(f"~/isaacsim/spot_lerobot_data/{current_date}")
        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # 태스크 정의 (속도 고정, 시간 조정)
        self.tasks = [
            {"name": "forward_walk", "command": np.array([2.0, 0.0, 0.0]), "task_index": 0, "duration": 2.5},
            {"name": "backward_walk", "command": np.array([-2.0, 0.0, 0.0]), "task_index": 1, "duration": 2.5},
            {"name": "left_walk", "command": np.array([0.0, 2.0, 0.0]), "task_index": 2, "duration": 2.5},
            {"name": "right_walk", "command": np.array([0.0, -2.0, 0.0]), "task_index": 3, "duration": 2.5},
            {"name": "rotate_left", "command": np.array([0.0, 0.0, 2.0]), "task_index": 4, "duration": 3.14},
            {"name": "rotate_right", "command": np.array([0.0, 0.0, -2.0]), "task_index": 5, "duration": 3.14},
            {"name": "forward_diagonal_left_walk", "command": np.array([2.0, 2.0, 0.0]), "task_index": 6, "duration": 2.5},
            {"name": "forward_diagonal_right_walk", "command": np.array([2.0, -2.0, 0.0]), "task_index": 7, "duration": 2.5},
        ]
        
        self.current_task = None
        self.current_episode_in_task = 0
        self.all_tasks_completed = False  # 모든 태스크 완료 플래그
        self.task_episode_count = 0  # 각 태스크별 에피소드 카운터 (0-99)
        
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
        
        # USD 스테이지 가져오기
        stage = omni.usd.get_context().get_stage()
        
        # DomeLight 생성
        dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
        
        # 조도 강도 설정
        dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(450.0)
        
        print("DomeLight가 성공적으로 추가되었습니다.")
        
    def setup_camera_on_body(self):
        """Spot의 body에 카메라를 부착"""
        print("Spot 로봇의 body에 카메라 부착 중...")
        
        camera_name = "/World/Spot/body/camera_rig"
        
        # Xform 프림 생성
        camera_prim = prim_utils.create_prim(
            camera_name,
            "Xform",
            translation=np.array([0.3, 0.0, 0.2]),
            orientation=euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
        )
        
        # 카메라 생성
        self.spot_camera = Camera(
            prim_path=f"{camera_name}/camera",
            frequency=25,
            resolution=(640, 480),
        )
        
        print(f"카메라가 {camera_name}/camera 경로에 생성되었습니다.")
        
    def start_new_task(self):
        """새로운 태스크 시작"""
        if self.current_task is None:
            self.current_task = self.tasks[0]
        else:
            # 다음 태스크로 이동
            current_task_idx = next(i for i, task in enumerate(self.tasks) if task['task_index'] == self.current_task['task_index'])
            if current_task_idx + 1 < len(self.tasks):
                self.current_task = self.tasks[current_task_idx + 1]
            else:
                # 모든 태스크 완료
                self.all_tasks_completed = True
                return False
                
        self.current_episode_in_task = 0
        self.task_episode_count = 0  # 태스크가 바뀔 때 에피소드 카운터 리셋
        print(f"새로운 태스크 시작: {self.current_task['name']} (task_index: {self.current_task['task_index']})")
        return True
        
    def start_new_episode(self):
        """새로운 에피소드 시작"""
        if self.current_episode_in_task >= self.episodes_per_task:
            # 현재 태스크의 모든 에피소드 완료
            if not self.start_new_task():
                return False  # 모든 태스크 완료
                
        # 현재 태스크의 지속시간에 따라 에피소드 길이 계산
        task_duration = self.current_task.get('duration', 5.0)  # 기본값 5초
        self.episode_length = int(task_duration / 0.02)  # 0.02초 간격으로 프레임 수 계산
        
        # 에피소드 데이터 초기화
        self.episode_data = {
            'observation.state': [],
            'observation.images': [],
            'action': [],
            'episode_index': [],
            'frame_index': [],
            'timestamp': [],
            'next.done': [],
            'index': [],
            'task_index': []
        }
        
        self.frame_count = 0
        
        print(f"새로운 에피소드 시작: {self.current_task['name']} (task_index: {self.current_task['task_index']}, episode_index: {self.task_episode_count}, index(global): {self.episode_count}, duration: {task_duration}s, frames: {self.episode_length})")
        return True
        
    def collect_robot_state(self):
        """로봇 상태 수집 (Spot의 _compute_observation과 동일한 48차원 구조)"""
        try:
            # 기본 상태 정보
            lin_vel_I = self.spot.robot.get_linear_velocity()
            ang_vel_I = self.spot.robot.get_angular_velocity()
            pos_IB, q_IB = self.spot.robot.get_world_pose()
            joint_pos = self.spot.robot.get_joint_positions()
            joint_vel = self.spot.robot.get_joint_velocities()
            
            # body frame으로 변환
            from isaacsim.core.utils.rotations import quat_to_rot_matrix
            R_IB = quat_to_rot_matrix(q_IB)
            R_BI = R_IB.transpose()
            
            lin_vel_b = np.matmul(R_BI, lin_vel_I)
            ang_vel_b = np.matmul(R_BI, ang_vel_I)
            gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))
            
            # 명령 (3차원)
            command = self.base_command
            
            # 기본 관절 위치 (Spot의 default pose)
            default_pos = np.array([0.0, 0.8, -1.6, 0.0, 0.8, -1.6, 0.0, 0.8, -1.6, 0.0, 0.8, -1.6])
            
            # 이전 액션 (초기화)
            if not hasattr(self, 'previous_action'):
                self.previous_action = np.zeros(12)
            
            # 48차원 관찰 벡터 구성 (SpotFlatTerrainPolicy와 동일)
            obs = np.zeros(48)
            obs[:3] = lin_vel_b      # body frame 선속도
            obs[3:6] = ang_vel_b     # body frame 각속도
            obs[6:9] = gravity_b     # body frame 중력
            obs[9:12] = command      # 명령
            obs[12:24] = joint_pos - default_pos  # 관절 위치 오프셋
            obs[24:36] = joint_vel   # 관절 속도
            obs[36:48] = self.previous_action  # 이전 액션
            
            return obs.astype(np.float32)
            
        except Exception as e:
            print(f"로봇 상태 수집 중 오류: {e}")
            return np.zeros(48, dtype=np.float32)
            
    def collect_action(self):
        """액션 데이터 수집 (12차원 관절 위치 명령)"""
        try:
            # 현재 액션 (12차원 관절 위치 명령)
            action = self.spot.action if hasattr(self.spot, 'action') else np.zeros(12)
            # 이전 액션 업데이트 (다음 관찰에서 사용)
            self.previous_action = action.copy()
            return action.astype(np.float32)
            
        except Exception as e:
            print(f"액션 수집 중 오류: {e}")
            return np.zeros(12, dtype=np.float32)
            
    def safe_capture_camera_image(self):
        """안전한 카메라 이미지 캡처 (6.spot_camera.py 방식)"""
        if self.spot_camera is not None and self.camera_ready:
            try:
                # RGB 이미지 가져오기
                rgb_data = self.spot_camera.get_rgba()
                if rgb_data is not None:
                    # 데이터 형태 확인
                    if hasattr(rgb_data, 'shape'):
                        if len(rgb_data.shape) == 3:
                            if rgb_data.shape[2] == 4:  # RGBA
                                rgb_image = rgb_data[:, :, :3]  # RGB로 변환
                            elif rgb_data.shape[2] == 3:  # RGB
                                rgb_image = rgb_data
                            else:
                                print(f"예상치 못한 채널 수: {rgb_data.shape[2]}")
                                return None
                        elif len(rgb_data.shape) == 2:  # 그레이스케일
                            rgb_image = np.stack([rgb_data] * 3, axis=2)
                        else:
                            print(f"예상치 못한 배열 차원: {len(rgb_data.shape)}")
                            return None
                    else:
                        print("rgb_data에 shape 속성이 없습니다")
                        return None
                    
                    # 이미지 파일명 생성 (episode_index와 일치하도록 수정)
                    image_filename = f"task{self.current_task['task_index']:02d}_ep{self.task_episode_count:02d}_frame{self.frame_count:03d}.png"
                    image_path = os.path.join(self.images_dir, image_filename)
                    
                    # PNG로 저장
                    plt.imsave(image_path, rgb_image)
                    
                    return image_filename
                    
            except Exception as e:
                print(f"카메라 이미지 캡처 중 오류 발생: {e}")
                return None
                
        return None
        
    def save_episode_data(self):
        """에피소드 데이터를 LeRobot 형식으로 저장"""
        if len(self.episode_data['observation.state']) == 0:
            return
            
        # DataFrame 생성
        df = pd.DataFrame(self.episode_data)
        
        # 파일명 생성 (2글자 episode 번호)
        filename = f"task{self.current_task['task_index']:02d}_ep{self.task_episode_count:02d}.parquet"
        filepath = os.path.join(self.output_dir, filename)
        
        # Parquet 형식으로 저장
        df.to_parquet(filepath, index=False)
        
        print(f"에피소드 {self.task_episode_count} 저장 완료: {filename}")
        print(f"  - 프레임 수: {len(self.episode_data['observation.state'])}")
        print(f"  - 태스크: {self.current_task['name']}")
        print(f"  - 에피소드 인덱스: {self.task_episode_count}")
        
    def create_metadata(self):
        """LeRobot 스키마에 맞는 메타데이터 생성"""
        metadata = {
            "codebase_version": "v2.0",
            "robot_type": "spot",
            "total_episodes": len(self.tasks) * self.episodes_per_task,
            "total_frames": len(self.tasks) * self.episodes_per_task * self.episode_length,
            "total_tasks": len(self.tasks),
            "total_videos": len(self.tasks) * self.episodes_per_task,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": 50,  # 0.02초 간격 = 50Hz
            "splits": {
                "train": f"0:{len(self.tasks) * self.episodes_per_task}"
            },
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [48],
                    "names": {
                        "body_velocities": ["lin_vel_x", "lin_vel_y", "lin_vel_z", "ang_vel_x", "ang_vel_y", "ang_vel_z"],
                        "gravity": ["gravity_x", "gravity_y", "gravity_z"],
                        "command": ["cmd_vx", "cmd_vy", "cmd_wz"],
                        "joint_offsets": [f"joint_offset_{i}" for i in range(12)],
                        "joint_velocities": [f"joint_vel_{i}" for i in range(12)],
                        "previous_action": [f"prev_action_{i}" for i in range(12)]
                    }
                },
                "observation.images": {
                    "dtype": "string",
                    "shape": [1],
                    "names": ["image_filename"]
                },
                "action": {
                    "dtype": "float32",
                    "shape": [12],
                    "names": {
                        "joint_positions": [f"joint_pos_{i}" for i in range(12)]
                    }
                },
                "episode_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "frame_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "timestamp": {
                    "dtype": "float32",
                    "shape": [1],
                    "names": None
                },
                "next.done": {
                    "dtype": "bool",
                    "shape": [1],
                    "names": None
                },
                "index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "task_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                }
            }
        }
        
        # 메타데이터 파일 저장
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        print(f"메타데이터 저장 완료: {metadata_path}")
        return metadata
        
    def on_physics_step(self, step_size):
        """물리 스텝 콜백"""
        if self.first_step:
            self.spot.initialize()
            self.spot_camera.initialize()
            self.first_step = False
            if not self.start_new_task():
                print("태스크 시작 실패")
                return
            if not self.start_new_episode():
                print("에피소드 시작 실패")
                return
            print("Spot과 카메라 초기화 완료")
            
        else:
            # 카메라 워밍업 처리
            if not self.camera_ready:
                if self.step_count >= self.camera_warmup_steps:
                    self.camera_ready = True
                    print("카메라 워밍업 완료")
                else:
                    # 워밍업 중에는 카메라 데이터만 가져와서 초기화
                    try:
                        if self.spot_camera is not None:
                            self.spot_camera.get_rgba()
                    except:
                        pass
                    self.step_count += 1
                    return
            
            # 명령 설정 (현재 태스크의 명령)
            self.base_command = self.current_task['command']
            
            # Spot 로봇 제어
            self.spot.forward(step_size, self.base_command)
            
            # 데이터 수집 (매 10스텝마다 = 0.02초)
            self.step_count += 1
            if self.step_count % self.data_collection_interval == 0:
                # 로봇 상태와 액션 수집
                robot_state = self.collect_robot_state()
                action = self.collect_action()
                image_filename = self.safe_capture_camera_image()
                
                # LeRobot 데이터 형식으로 저장
                is_last_frame = (self.frame_count == self.episode_length - 1)
                
                self.episode_data['observation.state'].append(robot_state)
                self.episode_data['observation.images'].append(image_filename)
                self.episode_data['action'].append(action)
                self.episode_data['episode_index'].append(self.task_episode_count)
                self.episode_data['frame_index'].append(self.frame_count)
                self.episode_data['timestamp'].append(self.frame_count * 0.02)  # 0.02초 간격
                self.episode_data['next.done'].append(is_last_frame)
                self.episode_data['index'].append(self.global_index)
                self.episode_data['task_index'].append(self.current_task['task_index'])
                
                self.frame_count += 1
                self.global_index += 1
                    
            # 에피소드 완료 체크 (250 프레임 = 5초)
            if self.frame_count >= self.episode_length:
                self.save_episode_data()
                self.episode_count += 1  # 글로벌 에피소드 카운터
                self.current_episode_in_task += 1
                self.task_episode_count += 1  # 태스크별 에피소드 카운터
                
                if not self.start_new_episode():
                    print("모든 데이터 수집 완료!")
                    self.all_tasks_completed = True
                    return
                
    def main(self):
        """메인 실행 함수"""
        print("Spot LeRobot 데이터 수집 파이프라인 시작")
        print(f"데이터 수집 간격: {self.data_collection_interval} 스텝 (0.02초)")
        print(f"태스크당 에피소드 수: {self.episodes_per_task}")
        print(f"총 태스크 수: {len(self.tasks)}")
        print(f"카메라 워밍업 스텝: {self.camera_warmup_steps}")
        print(f"출력 디렉토리: {self.output_dir}")
        print("\n태스크별 설정:")
        for task in self.tasks:
            duration = task.get('duration', 5.0)
            frames = int(duration / 0.02)
            print(f"  - {task['name']}: {duration}s ({frames} frames)")
        
        # World 설정
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 500.0,  # 0.002초
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
        total_episodes = len(self.tasks) * self.episodes_per_task  # 8 * 5 = 40개 에피소드
        max_steps = total_episodes * self.episode_length * self.data_collection_interval
        step_count = 0
        
        while simulation_app.is_running() and step_count < max_steps and not self.all_tasks_completed:
            self.world.step(render=True)
            step_count += 1
            
            # 주기적으로 상태 출력
            if step_count % 10000 == 0:
                print(f"진행 중... {step_count} 스텝, {self.episode_count}/{total_episodes} 에피소드 완료")
                
        print(f"데이터 수집 완료 - 총 {self.episode_count}개 에피소드 수집됨")
        print(f"저장된 데이터: {self.output_dir}")
        print(f"저장된 이미지: {self.images_dir}")
        
        # 메타데이터 생성 및 저장
        self.create_metadata()
        
        simulation_app.close()

if __name__ == "__main__":
    lerobot_collector = LeRobotSpotDataCollector()
    lerobot_collector.main()