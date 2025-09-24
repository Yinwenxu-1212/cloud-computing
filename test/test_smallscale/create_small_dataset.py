#!/usr/bin/env python3
"""
创建小规模测试数据集的脚本
从原始数据中选择：
- 10个客户群
- 30个数据中心
- 10月1日的数据（288个时间点）
"""

import pandas as pd
import numpy as np
import os

def create_small_dataset():
    """创建小规模测试数据集"""
    
    print("🚀 开始创建小规模测试数据集...")
    
    # 创建输出目录
    output_dir = "test_smallscale"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 创建目录: {output_dir}")
    
    # 1. 读取原始数据
    print("\n📖 读取原始数据...")
    demand_df = pd.read_csv("data/demand.csv")
    qos_df = pd.read_csv("data/qos.csv")
    bandwidth_df = pd.read_csv("data/site_bandwidth.csv")
    
    print(f"原始数据规模:")
    print(f"  - 需求数据: {demand_df.shape[0]} 时间点, {demand_df.shape[1]-1} 客户群")
    print(f"  - QoS数据: {qos_df.shape[0]} 数据中心, {qos_df.shape[1]-1} 客户群")
    print(f"  - 带宽数据: {bandwidth_df.shape[0]} 数据中心")
    
    # 2. 选择客户群（前10个：A-J）
    selected_customers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    print(f"\n🎯 选择客户群: {selected_customers}")
    
    # 3. 选择数据中心（前30个）
    selected_sites = qos_df['site_name'].head(30).tolist()
    print(f"🎯 选择数据中心: {len(selected_sites)} 个")
    print(f"   数据中心列表: {selected_sites}")
    
    # 4. 提取10月1日的数据（前288个时间点，即一天的数据）
    print(f"\n📅 提取10月1日数据（前288个时间点）...")
    
    # 创建小规模demand.csv
    demand_columns = ['mtime'] + selected_customers
    small_demand = demand_df[demand_columns].head(288).copy()
    
    print(f"小规模需求数据: {small_demand.shape[0]} 时间点, {len(selected_customers)} 客户群")
    
    # 5. 创建小规模qos.csv
    qos_columns = ['site_name'] + selected_customers
    small_qos = qos_df[qos_df['site_name'].isin(selected_sites)][qos_columns].copy()
    
    print(f"小规模QoS数据: {small_qos.shape[0]} 数据中心, {len(selected_customers)} 客户群")
    
    # 6. 创建小规模site_bandwidth.csv
    small_bandwidth = bandwidth_df[bandwidth_df['site_name'].isin(selected_sites)].copy()
    
    print(f"小规模带宽数据: {small_bandwidth.shape[0]} 数据中心")
    
    # 7. 保存文件
    print(f"\n💾 保存小规模数据集到 {output_dir}/ ...")
    
    small_demand.to_csv(f"{output_dir}/demand.csv", index=False)
    print(f"✅ 保存: {output_dir}/demand.csv")
    
    small_qos.to_csv(f"{output_dir}/qos.csv", index=False)
    print(f"✅ 保存: {output_dir}/qos.csv")
    
    small_bandwidth.to_csv(f"{output_dir}/site_bandwidth.csv", index=False)
    print(f"✅ 保存: {output_dir}/site_bandwidth.csv")
    
    # 8. 创建配置文件
    config_content = f"""[DEFAULT]
# 小规模测试数据集配置
customers = {len(selected_customers)}
sites = {len(selected_sites)}
time_slots = {small_demand.shape[0]}

[CUSTOMERS]
names = {','.join(selected_customers)}

[SITES]
count = {len(selected_sites)}
names = {','.join(selected_sites)}

[TIME]
start_date = 2021-10-01T00:00
end_date = 2021-10-01T23:55
interval_minutes = 5
total_slots = {small_demand.shape[0]}
"""
    
    with open(f"{output_dir}/config.ini", "w", encoding="utf-8") as f:
        f.write(config_content)
    print(f"✅ 保存: {output_dir}/config.ini")
    
    # 9. 数据统计报告
    print(f"\n📊 小规模数据集统计报告:")
    print(f"=" * 50)
    print(f"客户群数量: {len(selected_customers)}")
    print(f"数据中心数量: {len(selected_sites)}")
    print(f"时间点数量: {small_demand.shape[0]} (10月1日全天)")
    print(f"时间间隔: 5分钟")
    print(f"总变量数量估算: {len(selected_customers)} × {len(selected_sites)} × {small_demand.shape[0]} = {len(selected_customers) * len(selected_sites) * small_demand.shape[0]:,}")
    
    print(f"\n📈 需求数据统计:")
    for customer in selected_customers:
        daily_demand = small_demand[customer].sum()
        avg_demand = small_demand[customer].mean()
        max_demand = small_demand[customer].max()
        print(f"  {customer}: 日总需求={daily_demand:,.0f}, 平均={avg_demand:.1f}, 峰值={max_demand:,.0f}")
    
    print(f"\n🏢 数据中心带宽统计:")
    total_bandwidth = small_bandwidth['bandwidth'].sum()
    avg_bandwidth = small_bandwidth['bandwidth'].mean()
    print(f"  总带宽: {total_bandwidth:,}")
    print(f"  平均带宽: {avg_bandwidth:.0f}")
    print(f"  最小带宽: {small_bandwidth['bandwidth'].min():,}")
    print(f"  最大带宽: {small_bandwidth['bandwidth'].max():,}")
    
    print(f"\n🎯 数据集创建完成！")
    print(f"文件保存在: {output_dir}/")
    print(f"可以使用这个小规模数据集进行快速测试和调试。")
    
    return {
        'customers': len(selected_customers),
        'sites': len(selected_sites),
        'time_slots': small_demand.shape[0],
        'output_dir': output_dir
    }

if __name__ == "__main__":
    result = create_small_dataset()
    print(f"\n✨ 小规模数据集创建成功！")
    print(f"规模: {result['customers']}客户 × {result['sites']}站点 × {result['time_slots']}时间点")