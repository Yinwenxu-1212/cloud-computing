# Cloud Computing Research Project

## 项目概述

本项目专注于云计算中的带宽分配问题研究，包含相关文献、代码实现和研究成果。

## 目录结构

### 📚 literature/
包含与带宽分配问题相关的学术文献和研究论文：

- **核心论文**：
  - `2307.05975v1.pdf` - 相关研究论文
  - `Adaptive_Configuration_Selection_and_Bandwidth_Allocation_for_Edge-Based_Video_Analytics.pdf` - 边缘视频分析的自适应配置选择和带宽分配
  - `BandwidthAllocation_journal__2024_-12.pdf` - 带宽分配期刊论文
  - `CFR-RL_Traffic_Engineering_With_Reinforcement_Learning_in_SDN.pdf` - SDN中基于强化学习的流量工程
  - `nsdi21-singh Cost-Effective Cloud Edge Traffic Engineering with CASCARA .pdf` - CASCARA云边缘流量工程

- **DDD算法相关**：
  - `DDD/` 文件夹包含动态离散化发现(Dynamic Discretization Discovery)算法的相关文献
  - 涵盖连续时间服务网络设计、库存路由问题等应用

- **其他重要文献**：
  - 分层多智能体优化、深度强化学习在流量工程中的应用等

### 💻 test/
包含项目的核心代码实现：

- **主要算法**：
  - `Eadmm_test.py` - 扩展ADMM算法测试
  - `ddd_solver.py` - DDD求解器实现
  - `cplex_LP.py` / `cplex_MIP.py` - CPLEX线性规划和混合整数规划求解器

- **数据处理**：
  - `data/` - 包含需求数据、QoS数据、站点带宽数据
  - `load_data.py` - 数据加载模块

- **实验结果**：
  - `experiment_results/` - 实验结果和收敛性分析
  - `data_analysis/` - 数据分析报告和可视化图表

- **测试环境**：
  - `test_smallscale/` - 小规模测试数据集
  - `test_solution/` - 解决方案验证

### 📊 thought/
包含研究思路和展示材料：

- `DDD.pdf` - DDD算法相关理论
- `LBM.pdf` - 下界模型(Lower Bound Model)
- `UBM.pdf` - 上界模型(Upper Bound Model)  
- `带宽分配问题0918.pptx` - 项目展示PPT

## 研究内容

本项目主要研究云计算环境下的带宽分配优化问题，采用以下方法：

1. **ADMM算法**：分布式优化方法求解大规模带宽分配问题
2. **DDD算法**：动态离散化发现算法处理连续时间问题

## 技术栈

- **编程语言**：Python
- **优化求解器**：CPLEX
- **数据分析**：Pandas, NumPy, Matplotlib
- **机器学习**：强化学习框架

## 使用说明

1. 确保安装所需依赖包
2. 运行 `load_data.py` 加载数据
3. 执行相应的求解器进行优化计算
4. 查看 `experiment_results/` 中的结果分析

## 贡献者

本项目为云计算带宽分配问题的学术研究项目。

---

*最后更新：2025年*