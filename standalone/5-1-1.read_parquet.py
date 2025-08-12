#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import glob
from datetime import datetime

def find_latest_parquet():
    """가장 최근 parquet 파일 찾기"""
    base_path = os.path.expanduser("~/isaacsim/lerobot/spot")
    
    if not os.path.exists(base_path):
        print(f"경로가 존재하지 않습니다: {base_path}")
        return None
    
    # 모든 parquet 파일 찾기
    parquet_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    
    if not parquet_files:
        print("parquet 파일을 찾을 수 없습니다.")
        return None
    
    # 가장 최근 파일 선택
    latest_file = max(parquet_files, key=os.path.getctime)
    return latest_file

def read_and_display_parquet(file_path, num_rows=5):
    """parquet 파일을 읽어서 지정된 행 수만큼 출력"""
    try:
        print(f"파일 읽는 중: {file_path}")
        df = pd.read_parquet(file_path)
        
        print(f"\n=== 파일 정보 ===")
        print(f"총 행 수: {len(df)}")
        print(f"총 열 수: {len(df.columns)}")
        print(f"열 이름: {list(df.columns)}")
        
        print(f"\n=== 처음 {num_rows}행 ===")
        print(df.head(num_rows))
        
        print(f"\n=== 데이터 타입 ===")
        print(df.dtypes)
        
        print(f"\n=== 기본 통계 ===")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            print(df[numeric_columns].describe())
        
        return df
        
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return None

def analyze_robot_data(df):
    """로봇 데이터 분석"""
    if df is None or len(df) == 0:
        return
    
    print(f"\n=== 로봇 데이터 분석 ===")
    
    # 명령 데이터 분석
    if 'command' in df.columns:
        print("명령 데이터 샘플:")
        for i in range(min(3, len(df))):
            cmd = df.iloc[i]['command']
            print(f"  스텝 {i}: {cmd}")
    
    # 관절 위치 데이터 분석
    if 'joint_pos' in df.columns:
        print(f"\n관절 위치 데이터 (첫 3개 스텝):")
        for i in range(min(3, len(df))):
            joint_pos = df.iloc[i]['joint_pos']
            print(f"  스텝 {i}: {joint_pos[:3]}... (총 {len(joint_pos)}개 관절)")
    
    # 기본 속도 데이터 분석
    if 'base_vel' in df.columns:
        print(f"\n기본 선속도 데이터 (첫 3개 스텝):")
        for i in range(min(3, len(df))):
            base_vel = df.iloc[i]['base_vel']
            print(f"  스텝 {i}: {base_vel}")

def main():
    """메인 함수"""
    print("Spot 로봇 parquet 데이터 읽기")
    print("=" * 50)
    
    # 최신 parquet 파일 찾기
    latest_file = find_latest_parquet()
    
    if latest_file is None:
        print("사용 가능한 parquet 파일이 없습니다.")
        print("먼저 7.spot_lerobot_parquet.py를 실행하여 데이터를 생성하세요.")
        return
    
    print(f"발견된 파일: {latest_file}")
    
    # 파일 읽기 및 표시
    df = read_and_display_parquet(latest_file, num_rows=3)
    
    # 로봇 데이터 분석
    analyze_robot_data(df)
    
    print(f"\n" + "=" * 50)
    print("데이터 읽기 완료!")

if __name__ == "__main__":
    import numpy as np
    main()