#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:20:58 2026

@author: jaydenwang
"""

import numpy as np
rng = np.random.default_rng(42)


# 优先级 1: 投资组合与资产基础参数 (所有模型共用)

N = 100  # 金融机构/债务人的总数量

# u_i: 各个机构的总负债敞口 (Total liabilities)
u = np.ones(N)

# a_i, b_i: 各机构违约损失率 LGD (\epsilon_i) 的 Beta 分布参数
a = np.full(N, 2.0)
b = np.full(N, 5.0)

# p_i: 各机构的无条件边际违约概率 (Marginal Default Probability)
p = rng.uniform(0.01, 0.05, N)
p = p[np.argsort(p)]

# \rho_i: 各机构潜在资产价值对系统性风险因子 Z 的因子载荷 (Factor loadings)
# 当前设定采用 50% 的潜在变量资产相关度，即 rho_i^2=0.50。
# 如需 20% 的资产相关度，改为 np.sqrt(0.20)。
rho = np.full(N, np.sqrt(0.50))

# 异质性稳健性实验可显式替换为以下 PD 相关载荷；不要与基准同时启用：
# rho = 0.2 + 0.5 * (p - np.min(p)) / (np.max(p) - np.min(p))

# 优先级 2: Copula 模型依赖结构选择与特有参数

# [核心控制键] 可选值: 'Gaussian' 或 't' 或 'Skew-t'
# 更改此处参数后，所有主计算程序 (main, main_ineq, mc_verify) 将自动切换，无需手动注释代码
COPULA_TYPE = 'Skew-t'
XI = 2.0  # 依赖参数：XI = 1.0 退化为独立；XI > 1.0 开启 Gumbel 耦合

# [t 和 Skew-t 共用] 共同冲击因子 Lambda ~ Chi^2(nu) 的自由度
nu = 5.0  

# [Skew-t 特有] 系统因子 Z ~ Skew-Normal(0, 1, gamma) 的偏度/形状参数
# gamma > 0 代表右偏，在我们文章中的设定中Z越大代表宏观环境越退化
gamma = 0.5



# 优先级 3: 主程序控制、危机场景与风险度量置信水平

# main.py 读取这里的参数；方法和指标在 main.py 的 return 行选择。
CONDITION_MODE = "equality"  # "equality" 或 "inequality"
SHOW_PROGRESS = False  # 默认只显示最终结果与耗时；True 可另行开启进度条

# None: 按 alpha 自动求 VaR；数值: 所有方法使用同一个固定危机阈值。
# SYSTEM 对应机构损失 L_i；EXPOSURE 对应组合损失 L。
SYSTEM_DISTRESS_THRESHOLD = None
EXPOSURE_DISTRESS_THRESHOLD = None
# Exposure 自动阈值统一由 SPA 求得；benchmark 的 SE 不包含阈值误差。

target_i = 20  # 从 0 开始：200 表示第 201 个机构

alpha = 0.99  # 危机阈值置信水平：CoVaR 用个体 VaR；E-CoVaR 用组合 VaR
beta = 0.99   # 条件分位数置信水平：CoVaR / E-CoVaR 共用

# 优先级 4: SPA 精度与并行设置（也用于共用模型构建及组合 VaR）
SPA_GAUSSIAN_FACTOR_ORDER = 512
SPA_T_FACTOR_ORDER = 64       # t / Skew-t 的第一因子积分阶数
SPA_SCALE_ORDER = 64         # t / Skew-t 的尺度因子积分阶数
SPA_LGD_ORDER = 32
SPA_QUANTILE_TOL = 1e-5
SPA_SADDLE_BOUND = 20.0       # 必须在下面的 MGF 网格范围内
SPA_ROOT_TOL = 1e-10
SPA_MEAN_REGION = 1e-2
SPA_TAIL_BOUND_TOL = 1e-12

# 因子节点并行进程数。设为 1 可进行串行复核。
SPA_N_JOBS = 4
SPA_MGF_GRID_SIZE = 1_501
SPA_MGF_S_MIN = -25.0
SPA_MGF_S_MAX = 25.0


# 优先级 5: Plain Monte Carlo benchmark 参数

PMC_NUM_SIMS = 1_000_000     # 每次独立重复的路径数
PMC_NUM_RUNS = 10            # 独立重复次数，用于计算 standard error
PMC_BATCH_SIZE = 4_096       # 分批生成路径，控制内存占用
PMC_EQUALITY_BANDWIDTH_RATIO = 0.03  # CoVaR/CoES: h = ratio * u_i
PMC_EXPOSURE_EQUALITY_BANDWIDTH_RATIO = 0.03  # E-CoVaR/E-CoES: h = ratio * max(v, 1)
PMC_BASE_SEED = 42           # 第 r 次重复使用 BASE_SEED + r


# 优先级 6: CE-RQMC benchmark 参数

# Sobol 点数必须是 2 的整数次幂。
CE_RQMC_REPLICAS = 32
CE_RQMC_PILOT_SIZE = 8_192
CE_RQMC_FINAL_SIZE = 16_384

CE_RQMC_ELITE_FRACTION = 0.10
CE_RQMC_MAX_ITERATIONS = 5  # Hui Wang: iterative CE calibration uses at most five updates
# 单一的最终 equality bandwidth ratio；不再遍历多个 bandwidth ratios。
CE_RQMC_EQUALITY_BANDWIDTH_RATIO = 0.03  # CoVaR/CoES: h = ratio * u_i
CE_RQMC_EXPOSURE_EQUALITY_BANDWIDTH_RATIO = 0.03  # Exposure: h = ratio * max(v, 1)
# Pilot calibration may use a wider band, while the reported estimator always
# uses CE_RQMC_EQUALITY_BANDWIDTH_RATIO. Failed equality pilots widen this
# calibration band geometrically, bounded by CE_RQMC_MAX_PILOT_RETRIES.
CE_RQMC_CALIBRATION_BANDWIDTH_MULTIPLIER = 2.0
# Independent preliminary RQMC randomizations used to estimate conditional l0.
CE_RQMC_PRELIMINARY_SIZE = 8_192
CE_RQMC_PRELIMINARY_REPLICAS = 64
CE_RQMC_PRELIMINARY_BASE_SEED = 30_000
CE_RQMC_MAX_PILOT_RETRIES = 2
CE_RQMC_MIN_CALIBRATION_ESS = 64
CE_RQMC_MIN_ELITE_ESS = 5
CE_RQMC_PARAMETER_TOLERANCE = 0.02
CE_RQMC_POST_TARGET_UPDATES = 1
CE_RQMC_INITIAL_MU = 0.0
CE_RQMC_INITIAL_KAPPA = 1.0
CE_RQMC_MIN_KAPPA = 0.05
# 最终估计使用 delta*f_nominal + (1-delta)*g_CE；校准仍使用纯 CE。
# delta 越大越稳健但倾斜效率越弱。建议先使用 0.10，再做 0.05/0.10/0.20 敏感性分析。
CE_RQMC_DEFENSIVE_PROBABILITY = 0.10
CE_RQMC_PILOT_BASE_SEED = 10_000
CE_RQMC_BASE_SEED = 1_000

# 若 ESS × (1-beta) 低于此值，PMC/CE-RQMC 会发出精度警告。
BENCHMARK_MIN_EFFECTIVE_TAIL_SAMPLES = 20
