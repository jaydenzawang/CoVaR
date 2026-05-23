#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:58:34 2026

@author: jaydenwang
"""


import numpy as np
import time
import config
from copula import GaussianCopulaSingleFactor, TCopulaSingleFactor
from loss_model import PortfolioLoss
from var_calc import calculate_individual_var


np.random.seed(42)

def run_single_pmc(copula, portfolio, a_val_true, num_sims):
    """
    单次 Plain Monte Carlo 模拟，用于计算单个独立的 CoVaR 估计值
    通过直接计算减少中间大矩阵的建立，优化内存占用
    """
    # 模拟依赖结构中的潜变量并判定违约状态
    if isinstance(copula, GaussianCopulaSingleFactor):
        Z = np.random.normal(0, 1, size=(num_sims, 1))
        # 内存优化：直接生成潜在价值矩阵，避免单独存储大矩阵 eta
        latent = config.rho * Z + np.sqrt(1 - config.rho**2) * np.random.normal(0, 1, size=(num_sims, config.N))
        defaults = latent > copula.x_d
    elif isinstance(copula, TCopulaSingleFactor):
        Z = np.random.normal(0, 1, size=(num_sims, 1))
        Lambda = np.random.chisquare(config.nu, size=(num_sims, 1))
        sqrt_term = np.sqrt(Lambda / config.nu)
        latent = config.rho * Z + np.sqrt(1 - config.rho**2) * np.random.normal(0, 1, size=(num_sims, config.N))
        defaults = latent > (copula.x_d * sqrt_term)
    else:
        raise ValueError("未识别的 Copula 类型。")

    # 条件状态筛选 (Rejection Sampling Step)
    target_default_mask = defaults[:, config.target_i]
    valid_indices = np.where(target_default_mask)[0]
    num_valid = len(valid_indices)
    
    if num_valid == 0:
        return None

    # 仅提取满足危机条件的场景矩阵
    defaults_cond = defaults[valid_indices, :]

    # 向量化生成特定场景下的 LGD (Beta 分布随机数)
    beta_samples = np.random.beta(config.a, config.b, size=(num_valid, config.N))
    
    # 计算系统总损失组合
    portfolio_losses = defaults_cond * beta_samples * config.u
    
    # 【核心锁步】：根据 CoVaR 定义，将目标机构的损失精准锁定为其无条件 VaR
    portfolio_losses[:, config.target_i] = a_val_true

    # 横向求和得到各有效场景下的系统总损失
    total_system_losses = np.sum(portfolio_losses, axis=1)

    # 计算并返回当前批次的 CoVaR 分位数
    covar_est = np.percentile(total_system_losses, config.beta * 100)
    return covar_est


def evaluate_benchmark_with_se(num_sims, num_runs, copula):
    """
    多轮重复模拟实验，用于计算 CoVaR 的均值与标准误 (Standard Errors)
    """
    print("=" * 60)
    print(f"开始运行多轮 PMC 基准测试以计算标准误 (Standard Errors)...")
    print(f"每轮路径数 (num_sims): {num_sims} | 重复实验轮数 (num_runs): {num_runs}")
    print("=" * 60)


    portfolio = PortfolioLoss(config.u, config.a, config.b)
    
    print(f"当前评估依赖结构: {copula.__class__.__name__}")

    # 计算目标机构的无条件个体 VaR 作为基准线
    a_val_true = calculate_individual_var(portfolio, copula, config.target_i, config.alpha)
    print(f"机构 {config.target_i} 的无条件 VaR_{config.alpha} 基准值 = {a_val_true:.4f}\n")

    covar_results = []
    start_time = time.time()

    # 循环执行独立试验
    for run in range(1, num_runs + 1):
        run_start = time.time()
        covar_val = run_single_pmc(copula, portfolio, a_val_true, num_sims)
        
        if covar_val is not None:
            covar_results.append(covar_val)
            print(f"  [Run {run:02d}/{num_runs:02d}] 估计出的 CoVaR = {covar_val:.5f} | 耗时: {time.time() - run_start:.2f}秒")
        else:
            print(f"  [Run {run:02d}/{num_runs:02d}] 警告: 未能捕获有效违约场景！")

    # =============================================================================
    # 2. 统计指标计算 (Mean, Sample SD, Standard Error)
    # =============================================================================
    covar_results = np.array(covar_results)
    mean_covar = np.mean(covar_results)
    
    # 采用无偏样本标准差 (ddof=1)
    sd_covar = np.standard_deviation = np.std(covar_results, ddof=1)
    
    # 标准误公式：样本标准差 / 根号下重复实验次数
    se_covar = sd_covar / np.sqrt(len(covar_results))
    
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"PMC 终期统计验证报告 (置信度 beta={config.beta}):")
    print("-" * 60)
    print(f"CoVaR 期望均值 (Mean)         = {mean_covar:.5f}")
    print(f"估计值样本标准差 (SD)        = {sd_covar:.5f}")
    print(f"估计值标准误 (Standard Error) = {se_covar:.5f}  <-- 核心评价指标")
    print(f"95% 模拟统计置信区间         = [{mean_covar - 1.96 * se_covar:.5f}, {mean_covar + 1.96 * se_covar:.5f}]")
    print(f"总计耗时                     = {total_time:.2f} 秒")
    print("=" * 60)
    
    return mean_covar, se_covar

if __name__ == "__main__":
    
    # 模块化配置读取 (支持随时切换高斯或t-Copula)
    copula = GaussianCopulaSingleFactor(config.rho, config.p)
    # copula = TCopulaSingleFactor(config.rho, config.p, config.nu)
    
    # 执行 10 次重复试验，每次 500,000 条路径
    evaluate_benchmark_with_se(1_000_000, 10, copula)














