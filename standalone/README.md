# Isaac Sim Standalone Examples

이 디렉토리는 Isaac Sim을 사용한 다양한 로봇 시뮬레이션 예제들을 포함합니다.

## 📁 파일 목록

### 🤖 UR10 로봇 예제
- **`1.ur10_auto_pickandplace.py`**: UR10 로봇을 사용한 자동 Pick & Place 시뮬레이션

### 🤖 Franka 로봇 예제
- **`2.franka_replicate_video.py`**: Franka 로봇의 End Effector 카메라를 사용한 영상 저장 (Replicator 사용)
- **`3.franka_auto_pickandplace.py`**: Franka 로봇을 사용한 자동 Pick & Place 시뮬레이션
- **`4.franka_pickandplace_video.py`**: Franka 로봇의 Pick & Place 작업을 End Effector 카메라로 녹화

### 📊 데이터 분석 예제
- **`5-1-1.read_parquet.py`**: Parquet 파일을 읽어서 로봇 데이터를 분석하는 도구
- **`5-1.spot_lerobot_parquet.py`**: Spot 로봇의 키보드 제어 데이터를 LeRobot 형식의 Parquet 파일로 저장

### 🦿 Spot 로봇 예제
- **`5.spot_teleop_standalone.py`**: Spot 로봇의 키보드 원격 조작 (텔레오퍼레이션)
- **`6-1.spot_use_usd.py`**: 사용자 정의 USD 환경에서 Spot 로봇과 카메라 사용
- **`6.spot_camera.py`**: Spot 로봇의 body에 카메라를 부착하여 이미지 캡처

### 📈 데이터 수집 파이프라인
- **`7.spot_auto_datacollector.py`**: Spot 로봇의 자동 데이터 수집 (LeRobot 스키마 준수)
- **`7-1.spot_manual_datacollector.py`**: Spot 로봇의 수동 데이터 수집 (키보드 제어)
- **`7-2.spot_continuous_sampler.py`**: Spot 로봇의 연속 샘플링 데이터 수집 (고급 기능)

### 🔍 데이터 분석 도구
- **`8.spot_data_analyzer.py`**: Spot LeRobot 데이터의 Parquet 파일을 분석하는 도구
