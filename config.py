#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:20:58 2026

@author: jaydenwang
"""

import numpy as np
np.random.seed(42)


# 优先级 1: 投资组合与资产基础参数 (所有模型共用)

N = 1000  # 金融机构/债务人的总数量

# u_i: 各个机构的总负债敞口 (Total liabilities)
u = np.ones(N)

# a_i, b_i: 各机构违约损失率 LGD (\epsilon_i) 的 Beta 分布参数
a = np.full(N, 2.0)
b = np.full(N, 5.0)

# p_i: 各机构的无条件边际违约概率 (Marginal Default Probability)
p = np.random.uniform(0.01, 0.05, N)

# \rho_i: 各机构潜在资产价值对系统性风险因子 Z 的因子载荷 (Factor loadings)
rho = np.random.uniform(0.3, 0.6, N)



# 优先级 2: Copula 模型依赖结构选择与特有参数

# 【核心控制键】可选值: 'Gaussian' 或 't' 或 'Skew-t'
# 更改此处参数后，所有主计算程序 (main, main_ineq, mc_verify) 将自动切换，无需手动注释代码
COPULA_TYPE = 'Gaussian'
XI = 1.0  # 新增依赖参数：XI = 1.0 退化为独立；XI > 1.0 开启 Gumbel 耦合

# [t 和 Skew-t 共用] 共同冲击因子 Lambda ~ Chi^2(nu) 的自由度
nu = 5.0  

# [Skew-t 特有] 系统因子 Z ~ Skew-Normal(0, 1, gamma) 的偏度/形状参数
# gamma < 0 代表宏观环境左偏（恶化频率高，尾部更肥）；gamma > 0 代表右偏
gamma = -1.5  



# 优先级 3: 危机场景与风险度量置信水平

target_i = 0  # 陷入危机的目标金融机构索引 (例如第 0 个机构)

alpha = 0.99  # 用于计算目标机构个体 VaR 的置信水平 (如 0.99 或 0.999)
beta = 0.99   # 用于计算整个系统 CoVaR 的置信水平 (如 0.99 或 0.95)