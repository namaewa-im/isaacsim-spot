#!/usr/bin/env python3
"""
Spot LeRobot 데이터 분석기
Parquet 파일을 읽어서 열 정보와 데이터 샘플을 출력합니다.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

class SpotDataAnalyzer:
    def __init__(self, data_dir="spot_lerobot_data/20250811"):
        self.data_dir = Path(data_dir)
        
    def analyze_parquet_file(self, filename, num_rows=10):
        """Parquet 파일을 분석하여 정보를 출력합니다."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"파일을 찾을 수 없습니다: {filepath}")
            return
            
        print(f"=== {filename} 분석 ===")
        print(f"파일 경로: {filepath}")
        print(f"파일 크기: {filepath.stat().st_size / 1024:.1f} KB")
        print()
        
        # Parquet 파일 읽기
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            print(f"파일 읽기 오류: {e}")
            return
            
        # 기본 정보 출력
        print("📊 기본 정보:")
        print(f"  - 행 수: {len(df):,}")
        print(f"  - 열 수: {len(df.columns)}")
        print(f"  - 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        print()
        
        # 열 정보 출력
        print("📋 열 정보:")
        for i, col in enumerate(df.columns):
            dtype = df[col].dtype
            non_null = df[col].count()
            null_count = len(df) - non_null
            
            print(f"  {i+1:2d}. {col:20s} | {str(dtype):12s} | {non_null:6d} non-null | {null_count:3d} null")
            
            # 첫 번째 값 샘플 출력
            if non_null > 0:
                first_val = df[col].iloc[0]
                if isinstance(first_val, (list, np.ndarray)):
                    print(f"       샘플: {type(first_val).__name__} 길이 {len(first_val)}")
                else:
                    print(f"       샘플: {first_val}")
        print()
        
        # 데이터 샘플 출력
        print(f"📄 데이터 샘플 (처음 {num_rows}행):")
        print(df.head(num_rows).to_string())
        print()
        
        # 통계 정보 (수치형 열만)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("📈 수치형 열 통계:")
            print(df[numeric_cols].describe())
            print()
            
        # 특별한 열들 분석
        self._analyze_special_columns(df)
        
    def _analyze_special_columns(self, df):
        """특별한 열들을 자세히 분석합니다."""
        print("🔍 특별한 열 분석:")
        
        # observation 열 분석
        if 'observation' in df.columns:
            obs_data = df['observation'].iloc[0]
            if isinstance(obs_data, list):
                print(f"  - observation: {len(obs_data)}차원 벡터")
                print(f"    샘플: {obs_data[:5]}... (처음 5개 값)")
                
        # action 열 분석
        if 'action' in df.columns:
            action_data = df['action'].iloc[0]
            if isinstance(action_data, list):
                print(f"  - action: {len(action_data)}차원 벡터")
                print(f"    샘플: {action_data[:5]}... (처음 5개 값)")
                
        # command 열 분석
        if 'command' in df.columns:
            cmd_data = df['command'].iloc[0]
            if isinstance(cmd_data, list):
                print(f"  - command: {len(cmd_data)}차원 벡터 (vx, vy, wz)")
                print(f"    샘플: {cmd_data}")
                
        # timestamp 분석
        if 'timestamp' in df.columns:
            timestamps = df['timestamp']
            print(f"  - timestamp: {timestamps.min():.3f}s ~ {timestamps.max():.3f}s")
            print(f"    간격: {timestamps.diff().mean():.3f}s (평균)")
            
        # episode_index 분석
        if 'episode_index' in df.columns:
            ep_indices = df['episode_index']
            print(f"  - episode_index: {ep_indices.min()} ~ {ep_indices.max()}")
            
        # task_index 분석
        if 'task_index' in df.columns:
            task_indices = df['task_index']
            print(f"  - task_index: {task_indices.min()} ~ {task_indices.max()}")
            
        print()
        
    def list_all_files(self):
        """모든 Parquet 파일 목록을 출력합니다."""
        parquet_files = list(self.data_dir.glob("*.parquet"))
        parquet_files.sort()
        
        print(f"📁 {self.data_dir} 폴더의 Parquet 파일 목록:")
        print(f"총 {len(parquet_files)}개 파일")
        print()
        
        for i, filepath in enumerate(parquet_files):
            size_kb = filepath.stat().st_size / 1024
            print(f"{i+1:2d}. {filepath.name:20s} | {size_kb:6.1f} KB")
            
        return parquet_files

def main():
    analyzer = SpotDataAnalyzer()
    
    # 모든 파일 목록 출력
    files = analyzer.list_all_files()
    print("\n" + "="*60 + "\n")
    
    # 첫 번째 파일 분석 (예시)
    if files:
        first_file = files[0].name
        print(f"첫 번째 파일 '{first_file}' 분석 중...")
        analyzer.analyze_parquet_file(first_file, num_rows=5)
        
        # 다른 파일도 분석하고 싶다면 아래 주석 해제
        # analyzer.analyze_parquet_file("task01_ep00.parquet", num_rows=3)
        # analyzer.analyze_parquet_file("task07_ep04.parquet", num_rows=3)

if __name__ == "__main__":
    main() 