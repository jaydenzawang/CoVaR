#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:20:58 2026

@author: jaydenwang
"""

import numpy as np

# 数值实验的基本假设参数
N = 1000  # 金融机构的数量

# u_i: 各个机构的总负债 (Total liabilities)
np.random.seed(42)
# u = np.random.uniform(10, 100, N)
u = np.ones(N)

# a_i, b_i: 违约损失率 LGD (\epsilon_i) 的 Beta 分布参数
a = np.full(N, 2.0)
b = np.full(N, 5.0)

# p_i: 历史数据估计出的无条件违约概率
p = np.random.uniform(0.01, 0.05, N)

# \rho_i: 潜在违约变量对系统性风险因子 Z 的因子载荷 (Factor loadings)
rho = np.random.uniform(0.3, 0.6, N)


# t-copula
nu = 5.0  # t-分布的自由度 (Degrees of freedom)

# 危机事件发生的目标机构 (例如第 0 个机构处于危机中)
target_i = 0

# CoVaR 及 VaR 的置信水平定义
alpha = 0.99  # 用于计算个体机构 VaR 的置信水平
beta = 0.99   # 用于计算系统 CoVaR 的置信水平