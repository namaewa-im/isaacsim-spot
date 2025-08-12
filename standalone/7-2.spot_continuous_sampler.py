#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import omni.usd
import os, json, time, random
import pandas as pd
from datetime import datetime

import carb
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats

class ContinuousSpotDataCollector:
    def __init__(self):
        # ====== 타이밍 설정 ======
        self.physics_dt = 1.0 / 500.0            # 물리 500 Hz
        self.rendering_dt = 10 * self.physics_dt # 렌더 50 Hz
        self.save_every_n = 10                   # 10 스텝마다 저장 → 50 Hz
        self.save_dt = self.physics_dt * self.save_every_n  # 0.02 s

        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.physics_dt,
            rendering_dt=self.rendering_dt
        )

        # ====== 출력 디렉토리 ======
        current_date = datetime.now().strftime("%Y%m%d")
        self.output_dir = os.path.expanduser(f"~/isaacsim/spot_lerobot_data/{current_date}")
        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        # ====== 수집/스케줄 ======
        self.step_count = 0
        self.global_index = 0
        self.sim_time = 0.0

        # 요구: task 순서/ID 고정
        self.tasks = [
            {"name": "forward_walk",          "task_index": 0, "type": "linear"},
            {"name": "backward",              "task_index": 1, "type": "linear"},
            {"name": "left",                  "task_index": 2, "type": "linear"},
            {"name": "right",                 "task_index": 3, "type": "linear"},
            {"name": "diagonal",              "task_index": 4, "type": "linear"},
            {"name": "rotate_clockwise",      "task_index": 5, "type": "angular"},
            {"name": "rotate_counterclockwise","task_index": 6, "type": "angular"},
        ]
        self.episodes_per_task = 10
        self.current_task_idx = 0
        self.episode_idx_in_task = 0
        self.all_done = False

        # ====== 샘플링 범위 ======
        self.duration_range = (0.8, 3.0)     # sec
        self.speed_range = (0.2, 0.8)        # m/s (선속도 크기)
        self.angular_speed_range = (0.2, 0.8)# rad/s
        self.ramp_time = 0.2                 # sec

        self.current_episode = None
        self.episode_data = None
        self.previous_action = np.zeros(12)

        self.setup_scene()

    def setup_scene(self):
        # ground
        self.world.scene.add_default_ground_plane(
            z_position=0, name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.2, dynamic_friction=0.2, restitution=0.01,
        )
        # light
        from pxr import Sdf
        stage = omni.usd.get_context().get_stage()
        dome = stage.DefinePrim("/World/DomeLight", "DomeLight")
        dome.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(450.0)
        # robot
        self.spot = SpotFlatTerrainPolicy(
            prim_path="/World/Spot", name="Spot", position=np.array([0,0,0.8])
        )
        # camera
        self.spot_camera = Camera(
            prim_path="/World/Spot/body/camera",
            frequency=50, resolution=(256,256),
            position=np.array([0.3,0,0.2]),
            orientation=euler_angles_to_quats(np.array([0,0,0])),
        )

    # === 스케줄러 ===
    def advance_schedule(self):
        self.episode_idx_in_task += 1
        if self.episode_idx_in_task >= self.episodes_per_task:
            self.current_task_idx += 1
            self.episode_idx_in_task = 0
            if self.current_task_idx >= len(self.tasks):
                self.all_done = True

    # === 에피소드 파라미터 ===
    def sample_episode_parameters(self):
        task = self.tasks[self.current_task_idx]
        # duration, speed 샘플
        duration = random.uniform(*self.duration_range)
        speed = random.uniform(*self.speed_range)

        if task["type"] == "linear":
            if task["name"] == "forward_walk":
                cmd = np.array([+speed, 0.0, 0.0])
            elif task["name"] == "backward":
                cmd = np.array([-speed, 0.0, 0.0])
            elif task["name"] == "left":
                cmd = np.array([0.0, +speed, 0.0])
            elif task["name"] == "right":
                cmd = np.array([0.0, -speed, 0.0])
            elif task["name"] == "diagonal":
                # 대각선: 결과 속도 크기가 speed가 되도록 성분 = speed/√2
                s = speed / np.sqrt(2.0)
                # 사분면 무작위(+/+) (+/-) (-/+) (-/-)
                sx = random.choice([+1.0, -1.0])
                sy = random.choice([+1.0, -1.0])
                cmd = np.array([sx*s, sy*s, 0.0])
            else:
                cmd = np.array([+speed, 0.0, 0.0])
        else:
            # yaw+: CCW, yaw-: CW
            amin, amax = self.angular_speed_range
            ang = random.uniform(amin, amax)
            if task["name"] == "rotate_counterclockwise":
                cmd = np.array([0.0, 0.0, +ang])   # CCW
            else:  # rotate_clockwise
                cmd = np.array([0.0, 0.0, -ang])   # CW

        return {
            "task": task,
            "duration": duration,
            "speed": speed,
            "command": cmd,
            "start_pose": None,
            "start_sim_time": None,
        }

    def start_new_episode(self):
        self.current_episode = self.sample_episode_parameters()
        # reset pose
        self.spot.robot.set_world_pose(
            position=np.array([0,0,0.8]),
            orientation=np.array([1,0,0,0])  # (w,x,y,z)
        )
        start_pos, start_quat = self.spot.robot.get_world_pose()
        self.current_episode["start_pose"] = (start_pos, start_quat)
        self.current_episode["start_sim_time"] = self.sim_time

        # 프레임별 저장 버퍼
        self.episode_data = {
            "observations": [], "actions": [],
            "episode_index": self.episode_idx_in_task,   # 각 task 내 0..9
            "frame_indices": [], "timestamps_s": [],
            "next_done": [], "indices_global": [],
            "task_index": self.current_episode["task"]["task_index"],
            "task_name": self.current_episode["task"]["name"],
            "command_base": self.current_episode["command"].tolist(),
            "command": [],
            "command_duration_s": [],
            "duration": self.current_episode["duration"],
            "speed": self.current_episode["speed"],
            "start_pose": self.current_episode["start_pose"],
            "achieved_delta": None,
        }

        print(f"[Task {self.current_episode['task']['task_index']}:{self.current_episode['task']['name']}] "
              f"Ep {self.episode_idx_in_task}/10  "
              f"T={self.current_episode['duration']:.2f}s  cmd={self.current_episode['command']}")

    # === 관측/액션 ===
    def collect_robot_state(self):
        # 안전한 상태 수집 (빈 배열 방지)
        try:
            lin_vel_I = self.spot.robot.get_linear_velocity()
            ang_vel_I = self.spot.robot.get_angular_velocity()
            pos_IB, q_IB = self.spot.robot.get_world_pose()
            
            # 빈 배열 체크 및 기본값 설정
            if len(lin_vel_I) == 0:
                lin_vel_I = np.zeros(3)
            if len(ang_vel_I) == 0:
                ang_vel_I = np.zeros(3)
            if len(pos_IB) == 0:
                pos_IB = np.zeros(3)
            if len(q_IB) == 0:
                q_IB = np.array([1.0, 0.0, 0.0, 0.0])
                
        except Exception as e:
            print(f"로봇 상태 수집 오류: {e}")
            # 기본값으로 초기화
            lin_vel_I = np.zeros(3)
            ang_vel_I = np.zeros(3)
            pos_IB = np.zeros(3)
            q_IB = np.array([1.0, 0.0, 0.0, 0.0])

        from isaacsim.core.utils.rotations import quat_to_rot_matrix
        R_IB = quat_to_rot_matrix(q_IB); R_BI = R_IB.transpose()
        lin_vel_b = R_BI @ lin_vel_I
        ang_vel_b = R_BI @ ang_vel_I
        gravity_b = R_BI @ np.array([0.0, 0.0, -1.0])

        # 관절 상태도 안전하게 수집
        try:
            joint_pos = self.spot.robot.get_joint_positions()
            joint_vel = self.spot.robot.get_joint_velocities()
        except:
            joint_pos = np.zeros(12)
            joint_vel = np.zeros(12)
            
        default_pos = np.array([0.0, 0.8, -1.6]*4)

        obs = np.zeros(48, dtype=float)
        obs[0:3]   = lin_vel_b
        obs[3:6]   = ang_vel_b
        obs[6:9]   = gravity_b
        obs[9:12]  = self.spot.base_command
        obs[12:24] = joint_pos - default_pos
        obs[24:36] = joint_vel
        obs[36:48] = getattr(self, "previous_action", np.zeros(12))
        return obs

    def collect_action_data(self):
        return self.spot.action if hasattr(self.spot, "action") and self.spot.action is not None else np.zeros(12)

    # === 델타/유틸 ===
    def quat_to_yaw(self, q):
        w,x,y,z = q
        return np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

    def calculate_achieved_delta(self):
        if not self.current_episode or not self.current_episode["start_pose"]:
            return None
        start_pos, start_quat = self.current_episode["start_pose"]
        cur_pos, cur_quat = self.spot.robot.get_world_pose()
        from isaacsim.core.utils.rotations import quat_multiply, quat_conjugate
        rot_delta = quat_multiply(cur_quat, quat_conjugate(start_quat))
        pos_delta = cur_pos - start_pos
        return {
            "pos_delta": pos_delta.tolist(),
            "rot_delta": rot_delta.tolist(),
            "distance": float(np.linalg.norm(pos_delta)),
            "yaw_change": float(self.quat_to_yaw(rot_delta)),
        }

    def apply_ramp_command(self, elapsed_time):
        duration = self.current_episode["duration"]
        base_cmd = self.current_episode["command"]
        ramp = min(self.ramp_time, duration/2.0)
        t = elapsed_time
        if t < 0.0: scale = 0.0
        elif t < ramp: scale = t / ramp
        elif t <= duration - ramp: scale = 1.0
        elif t <= duration: scale = max(0.0, (duration - t) / ramp)
        else: scale = 0.0
        return base_cmd * float(np.clip(scale, 0.0, 1.0))

    # === 저장 ===
    def save_episode_data(self):
        n = len(self.episode_data["observations"])
        if n == 0: return
        self.episode_data["achieved_delta"] = self.calculate_achieved_delta()

        df = pd.DataFrame({
            "observation": self.episode_data["observations"],
            "action": self.episode_data["actions"],
            "episode_index": [self.episode_data["episode_index"]]*n,   # 0..9 (task 내)
            "frame_index": self.episode_data["frame_indices"],         # 0..(n-1)
            "timestamp": self.episode_data["timestamps_s"],          # 0.00, 0.02, ...
            "next.done": self.episode_data["next_done"],
            "index": self.episode_data["indices_global"],               # 전역 인덱스
            "task_index": [self.episode_data["task_index"]]*n,
            "command": self.episode_data["command"],
            "command.duration_s": self.episode_data["command_duration_s"],
        })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"task{self.episode_data['task_index']:02d}_ep{self.episode_data['episode_index']:04d}_{timestamp}.parquet"
        df.to_parquet(os.path.join(self.output_dir, fname), index=False)

        meta = {
            "task_name": self.episode_data["task_name"],
            "task_index": self.episode_data["task_index"],
            "episode_index": self.episode_data["episode_index"],
            "frame_count": n,
            "duration": self.episode_data["duration"],
            "speed": self.episode_data["speed"],
            "command_base": self.episode_data["command_base"],
            "achieved_delta": self.episode_data["achieved_delta"],
            "timestamp": timestamp,
        }
        with open(os.path.join(self.output_dir, fname.replace(".parquet", "_meta.json")), "w") as f:
            json.dump(meta, f, indent=2)

        d = self.episode_data["achieved_delta"]
        print(f"저장: {fname} (frames={n}, dist={d['distance']:.3f}m, yaw={d['yaw_change']:.3f}rad)")

    # === 물리 콜백 ===
    def on_physics_step(self, step_size):
        self.sim_time += step_size
        if self.all_done:
            return

        if self.current_episode is None:
            self.start_new_episode()

        elapsed = self.sim_time - self.current_episode["start_sim_time"]
        cmd = self.apply_ramp_command(elapsed)
        self.spot.base_command = cmd
        
        # 로봇이 완전히 초기화되었는지 확인
        try:
            # 간단한 상태 확인
            test_vel = self.spot.robot.get_linear_velocity()
            if len(test_vel) == 0:
                return  # 아직 초기화되지 않음, 건너뛰기
            self.spot.forward(step_size, cmd)
        except Exception as e:
            print(f"로봇 제어 오류 (무시): {e}")
            return  # 오류 발생 시 건너뛰기

        # 저장 주기(50 Hz)마다 로깅
        if (self.step_count % self.save_every_n) == 0:
            frame_idx = self.episode_data["frame_indices"][-1] + 1 if self.episode_data["frame_indices"] else 0

            state = self.collect_robot_state()
            action = self.collect_action_data()

            self.episode_data["observations"].append(state.tolist())
            self.episode_data["actions"].append(action.tolist())
            self.episode_data["frame_indices"].append(frame_idx)
            # 요구: timestamp는 정확히 0.02*s로 증가
            self.episode_data["timestamps_s"].append(frame_idx * self.save_dt)
            self.episode_data["next_done"].append(False)
            self.episode_data["indices_global"].append(self.global_index)

            base_cmd = self.current_episode["command"]
            self.episode_data["command"].append(base_cmd.tolist())
            self.episode_data["command_duration_s"].append(float(self.current_episode["duration"]))

            self.previous_action = action.copy()
            self.global_index += 1

        # 에피소드 종료
        if elapsed >= self.current_episode["duration"]:
            if self.episode_data["next_done"]:
                self.episode_data["next_done"][-1] = True
            self.save_episode_data()
            self.current_episode = None
            self.advance_schedule()  # 다음 에피소드/다음 task로

        # 콜백 동작 확인용 로그 (매 500스텝마다)
        if (self.step_count % 500) == 0:
            print(f"[tick] sim_time={self.sim_time:.3f}")
            
        self.step_count += 1

    def run(self):
        self.world.reset()
        
        # 물리 시뮬레이션 안정화를 위한 대기
        print("물리 시뮬레이션 안정화 중...")
        for _ in range(500):  # 500 스텝 대기 (1초)
            self.world.step(render=False)  # 렌더링 없이 빠르게
        print("안정화 완료, 데이터 수집 시작")
        
        # 초기화 강제(뷰 생성 유도)
        _ = self.spot.robot.get_world_pose()
        
        # 안정화 후 physics 콜백 등록
        self.world.add_physics_callback("collector", self.on_physics_step)
        
        while simulation_app.is_running():
            if self.all_done:
                break
            self.world.step(render=True)
        simulation_app.close()

if __name__ == "__main__":
    collector = ContinuousSpotDataCollector()
    collector.run()
