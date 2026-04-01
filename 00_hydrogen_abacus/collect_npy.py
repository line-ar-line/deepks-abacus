#!/usr/bin/env python3
"""
检索并分类所有npy文件，按名称合并到npydata文件夹中

用法:
    python collect_npy.py                    # 使用当前目录作为搜索目录
    python collect_npy.py /path/to/search    # 指定搜索目录
    python collect_npy.py -h                 # 显示帮助信息
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
import re

class Logger:
    """同时输出到控制台和文件的日志类"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

def find_npy_files(root_dir):
    """递归查找所有npy文件，排除npydata目录"""
    npy_files = []
    for root, dirs, files in os.walk(root_dir):
        # 跳过npydata目录
        if 'npydata' in root:
            continue
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    return npy_files

def classify_npy_files(npy_files):
    """按文件名分类npy文件"""
    classified = {
        'atom': [],
        'box': [],
        'energy': []
    }
    
    for file_path in npy_files:
        basename = os.path.basename(file_path)
        # 匹配 deepks_atom.npy, deepks_box.npy, deepks_energy.npy
        if 'atom' in basename.lower():
            classified['atom'].append(file_path)
        elif 'box' in basename.lower():
            classified['box'].append(file_path)
        elif 'energy' in basename.lower():
            classified['energy'].append(file_path)
    
    return classified

def preview_data(data, name="数据", max_elements=20):
    """预览数据内容"""
    print(f"\n  === {name} 预览 ===")
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    
    # 根据维度决定如何显示
    total_elements = data.size
    if total_elements <= max_elements:
        print(f"  内容:\n{data}")
    else:
        print(f"  内容 (显示前{max_elements}个元素):")
        flat_data = data.flatten()
        preview = flat_data[:max_elements]
        print(f"  {preview}")
        if total_elements > max_elements:
            print(f"  ... (共 {total_elements} 个元素)")
    
    # 显示统计信息
    print(f"  统计: min={data.min():.6f}, max={data.max():.6f}, mean={data.mean():.6f}")

def merge_npy_files(file_list, output_path, category):
    """合并npy文件，提升一个维度"""
    if not file_list:
        print(f"警告: 没有文件需要合并到 {output_path}")
        return
    
    arrays = []
    print(f"\n  {'='*50}")
    print(f"  处理类别: {category}")
    print(f"  {'='*50}")
    
    for idx, file_path in enumerate(sorted(file_list)):
        try:
            data = np.load(file_path)
            arrays.append(data)
            print(f"\n  [文件 {idx+1}/{len(file_list)}] {os.path.basename(file_path)}")
            preview_data(data, f"原始数据 (shape: {data.shape})")
        except Exception as e:
            print(f"  错误: 无法加载 {file_path}: {e}")
    
    if not arrays:
        print(f"警告: 没有成功加载任何数组到 {output_path}")
        return
    
    # 检查所有数组的shape是否一致（除了第一维）
    base_shape = arrays[0].shape
    for i, arr in enumerate(arrays[1:], 1):
        if arr.shape != base_shape:
            print(f"  警告: 数组shape不一致: 文件{i+1}的shape为 {arr.shape}, 期望 {base_shape}")
    
    # 合并数组，增加一个维度
    merged = np.stack(arrays, axis=0)
    
    print(f"\n  {'='*50}")
    print(f"  合并结果")
    print(f"  {'='*50}")
    preview_data(merged, f"合并后数据 ({len(arrays)} 帧)")
    print(f"  维度提升: {base_shape} -> {merged.shape}")
    print(f"  新增维度 (帧数 n): {len(arrays)}")
    
    # 保存合并后的数组
    np.save(output_path, merged)
    print(f"\n  已保存到: {output_path}")
    print(f"  文件大小: {os.path.getsize(output_path) / 1024:.2f} KB")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='检索并分类所有npy文件，按名称合并到npydata文件夹中',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python collect_npy.py                          # 使用当前目录
  python collect_npy.py /path/to/data            # 指定搜索目录
  python collect_npy.py ./00_H_scf               # 使用相对路径
        '''
    )
    parser.add_argument(
        'search_dir',
        nargs='?',
        default='.',
        help='要搜索的目录路径 (默认: 当前目录)'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='输出目录路径 (默认: 在搜索目录下创建npydata)'
    )
    
    args = parser.parse_args()
    
    # 获取搜索目录的绝对路径
    root_dir = os.path.abspath(args.search_dir)
    
    # 检查搜索目录是否存在
    if not os.path.exists(root_dir):
        print(f"错误: 搜索目录不存在: {root_dir}")
        sys.exit(1)
    
    if not os.path.isdir(root_dir):
        print(f"错误: 指定的路径不是目录: {root_dir}")
        sys.exit(1)
    
    # 设置输出目录
    if args.output:
        npydata_dir = os.path.abspath(args.output)
    else:
        npydata_dir = os.path.join(root_dir, 'npydata')
    
    # 创建npydata目录（如果不存在）
    os.makedirs(npydata_dir, exist_ok=True)
    
    # 设置日志输出到OUTPUT文件
    output_log = os.path.join(npydata_dir, 'OUTPUT')
    logger = Logger(output_log)
    sys.stdout = logger
    
    try:
        print(f"搜索目录: {root_dir}")
        print(f"输出目录: {npydata_dir}")
        
        # 查找所有npy文件
        npy_files = find_npy_files(root_dir)
        print(f"\n找到 {len(npy_files)} 个npy文件:")
        for f in npy_files:
            print(f"  {f}")
        
        # 分类npy文件
        classified = classify_npy_files(npy_files)
        
        print(f"\n分类结果:")
        for category, files in classified.items():
            print(f"  {category}: {len(files)} 个文件")
        
        print(f"\n日志文件: {output_log}")
        
        # 合并并保存各类npy文件
        print("\n开始合并文件:")
        for category, files in classified.items():
            if files:
                output_path = os.path.join(npydata_dir, f'{category}.npy')
                merge_npy_files(files, output_path, category)
        
        # 最终汇总
        print("\n" + "="*60)
        print("处理完成! 生成的文件汇总:")
        print("="*60)
        for category in ['atom', 'box', 'energy']:
            output_path = os.path.join(npydata_dir, f'{category}.npy')
            if os.path.exists(output_path):
                data = np.load(output_path)
                print(f"\n{category}.npy:")
                print(f"  - 路径: {output_path}")
                print(f"  - Shape: {data.shape}")
                print(f"  - Dtype: {data.dtype}")
                print(f"  - 大小: {os.path.getsize(output_path) / 1024:.2f} KB")
        print("\n" + "="*60)
        
    finally:
        # 恢复标准输出并关闭日志文件
        sys.stdout = logger.terminal
        logger.close()
        print(f"\n输出已保存到: {output_log}")

if __name__ == '__main__':
    main()
