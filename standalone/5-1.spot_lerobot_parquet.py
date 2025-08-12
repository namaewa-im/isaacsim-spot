#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
import omni.appwindow  # Contains handle to keyboard
from isaacsim.core.api import World
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
import os
import pandas as pd
from datetime import datetime
from isaacsim.core.utils.rotations import quat_to_rot_matrix

class QuadrupedKeyboardStandalone:
    def __init__(self):
        self.world = None
        self.spot = None
        self.base_command = np.array([0.0, 0.0, 0.0])
        self.physics_ready = False
        self.first_step = True
        self.reset_needed = False
        
        # 데이터 수집 관련 변수들
        self.data_records = []
        self.step_count = 0
        self.last_action = np.zeros(12)
        
        # 저장 경로 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.expanduser(f"~/isaacsim/lerobot/spot/{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        self.parquet_path = os.path.join(self.save_dir, f"{timestamp}.parquet")
        
        print(f"데이터 저장 경로: {self.parquet_path}")
        
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
    
    def collect_robot_state(self):
        """로봇 상태 수집 (LeRobot 포맷)"""
        if not hasattr(self.spot, 'robot') or self.spot.robot is None:
            return None
            
        try:
            # 기본 속도 정보
            lin_vel_I = self.spot.robot.get_linear_velocity()
            ang_vel_I = self.spot.robot.get_angular_velocity()
            pos_IB, q_IB = self.spot.robot.get_world_pose()
            
            # 좌표계 변환
            R_IB = quat_to_rot_matrix(q_IB)
            R_BI = R_IB.transpose()
            lin_vel_b = np.matmul(R_BI, lin_vel_I)
            ang_vel_b = np.matmul(R_BI, ang_vel_I)
            gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))
            
            # 관절 상태
            joint_pos = self.spot.robot.get_joint_positions()
            joint_vel = self.spot.robot.get_joint_velocities()
            
            # LeRobot 포맷으로 상태 구성
            state = {
                "base_vel": lin_vel_b.tolist(),
                "base_ang_vel": ang_vel_b.tolist(),
                "gravity": gravity_b.tolist(),
                "joint_pos": joint_pos.tolist(),
                "joint_vel": joint_vel.tolist(),
                "command": self.base_command.tolist(),
                "previous_action": self.last_action.tolist(),
                "world_pos": pos_IB.tolist(),
                "world_quat": q_IB.tolist()
            }
            
            return state
            
        except Exception as e:
            print(f"상태 수집 중 오류: {e}")
            return None
    
    def collect_action_data(self):
        """액션 데이터 수집"""
        if not hasattr(self.spot, 'action') or self.spot.action is None:
            return None
            
        try:
            # 현재 액션 계산
            current_action = self.spot.default_pos + (self.spot.action * self.spot._action_scale)
            
            action_data = {
                "joint_pos": current_action.tolist(),
                "action_offset": self.spot.action.tolist(),
                "action_scale": self.spot._action_scale
            }
            
            # 이전 액션 업데이트
            self.last_action = self.spot.action.copy()
            
            return action_data
            
        except Exception as e:
            print(f"액션 수집 중 오류: {e}")
            return None
    
    def save_data_record(self, state, action):
        """데이터 레코드 저장"""
        if state is None or action is None:
            return
            
        record = {
            "step": self.step_count,
            "timestamp": datetime.now().isoformat(),
            **state,
            **action
        }
        
        self.data_records.append(record)
        
        # 주기적으로 파일에 저장 (메모리 절약)
        if len(self.data_records) % 100 == 0:
            self.save_to_parquet()
    
    def save_to_parquet(self):
        """데이터를 parquet 파일로 저장"""
        if not self.data_records:
            return
            
        try:
            df = pd.DataFrame(self.data_records)
            df.to_parquet(self.parquet_path, index=False)
            print(f"데이터 저장됨: {len(self.data_records)} 레코드 -> {self.parquet_path}")
        except Exception as e:
            print(f"파일 저장 중 오류: {e}")
        
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
            # 로봇 상태 수집
            state = self.collect_robot_state()
            
            # Spot 로봇 실행
            self.spot.forward(step_size, self.base_command)
            
            # 액션 데이터 수집
            action = self.collect_action_data()
            
            # 데이터 저장
            self.save_data_record(state, action)
            
            self.step_count += 1
            
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
        print(f"데이터 저장 경로: {self.save_dir}")
        
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
        
        try:
            while simulation_app.is_running() and step_count < max_steps:
                self.world.step(render=True)
                step_count += 1
                
                # 시뮬레이션이 중지되면 리셋 플래그 설정
                if self.world.is_stopped():
                    self.reset_needed = True
                    
                # 주기적으로 상태 출력
                if step_count % 500 == 0:
                    print(f"진행 중... {step_count} 스텝, 현재 명령: {self.base_command}")
                    print(f"수집된 데이터: {len(self.data_records)} 레코드")
                    
        except KeyboardInterrupt:
            print("\n사용자에 의해 중단됨")
        finally:
            # 최종 데이터 저장
            self.save_to_parquet()
            print(f"최종 데이터 저장 완료: {len(self.data_records)} 레코드")
            print(f"저장 위치: {self.parquet_path}")
            
        print("Spot 로봇 키보드 제어 완료")
        simulation_app.close()

if __name__ == "__main__":
    quadruped = QuadrupedKeyboardStandalone()
    quadruped.main()