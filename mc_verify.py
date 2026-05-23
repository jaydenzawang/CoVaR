#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:58:34 2026

@author: jaydenwang
"""


import numpy as np
import time
from scipy.stats import norm

# 导入我们已有的配置和依赖
import config
from copula import GaussianCopulaSingleFactor
from var_calc import calculate_individual_var

def monte_carlo_covar(beta, a_val, num_simulations=100000):
    """
    使用加权蒙特卡洛模拟计算 CoVaR
    """
    print(f"启动 Monte Carlo 模拟 (模拟次数: {num_simulations:,})...")
    start_time = time.time()
    
    # 实例化 Copula 模型
    copula = GaussianCopulaSingleFactor(config.rho, config.p)
    N = config.N
    target_i = config.target_i
    
    # 为了避免内存爆炸，我们将 10 万次模拟分批进行 (比如每批 10000 次)
    batch_size = 10000
    batches = num_simulations // batch_size
    
    all_system_losses = []
    all_weights = []
    
    for b in range(batches):
        # 1. 模拟系统性风险因子 Z ~ N(0, 1)
        Z_sim = np.random.standard_normal(batch_size)
        
        # 2. 批量计算所有机构的条件违约概率 p_{j|Z}
        # 结果 p_cond_matrix 的维度: (batch_size, N)
        # 我们巧妙利用 numpy 的外积/广播机制
        num = np.outer(Z_sim, config.rho) - copula.x_d
        den = np.sqrt(1 - config.rho**2)
        p_cond_matrix = norm.cdf(np.clip(num / den, -10, 10))
        
        # 提取目标机构 i 的条件违约概率作为权重 (贝叶斯后验权重)
        weights = p_cond_matrix[:, target_i]
        all_weights.append(weights)
        
        # 3. 模拟各机构的违约情况 D_j (0或1)
        # 使用随机数矩阵与概率矩阵比较，快速生成二项分布
        rand_unif = np.random.rand(batch_size, N)
        defaults = (rand_unif < p_cond_matrix).astype(float)
        
        # 4. 模拟 LGD (违约损失率) eps_j ~ Beta(a_j, b_j)
        # 为所有的机构生成 Beta 分布随机数
        # 注意：实际上只需要为违约的机构生成，但为了向量化计算速度，全量生成再相乘更快
        a_matrix = np.tile(config.a, (batch_size, 1))
        b_matrix = np.tile(config.b, (batch_size, 1))
        lgd_sim = np.random.beta(a_matrix, b_matrix)
        
        # 5. 计算损失 L_j = u_j * D_j * eps_j
        # 强制机构 i 的损失为 a_val (因为我们在做条件模拟 L_i = a)
        losses = config.u * defaults * lgd_sim
        losses[:, target_i] = a_val  
        
        # 6. 汇总系统的总损失
        system_loss = np.sum(losses, axis=1)
        all_system_losses.append(system_loss)
        
    # 拼接所有批次的数据
    all_system_losses = np.concatenate(all_system_losses)
    all_weights = np.concatenate(all_weights)
    
    # 7. 计算加权经验累积分布函数 (Weighted Empirical CDF)
    # 按系统损失从小到大排序
    sorted_indices = np.argsort(all_system_losses)
    sorted_losses = all_system_losses[sorted_indices]
    sorted_weights = all_weights[sorted_indices]
    
    # 归一化权重，使其总和为 1
    normalized_weights = sorted_weights / np.sum(sorted_weights)
    
    # 计算累计权重 (即经验 CDF)
    cumulative_weights = np.cumsum(normalized_weights)
    
    # 8. 找到第一个累积概率大于等于 beta 的损失值，即为 CoVaR
    covar_idx = np.searchsorted(cumulative_weights, beta)
    covar_mc = sorted_losses[covar_idx]
    
    elapsed = time.time() - start_time
    print(f"Monte Carlo 模拟完成，耗时: {elapsed:.2f} 秒")
    
    return covar_mc

if __name__ == "__main__":
    
    # 开始计时
    start_time = time.time()
    
    # 如果您在上一步把 u 设为了全是 1，为了确保我们拿到有效的 a_val，这里用 alpha=0.99
    alpha_val = 0.99
    beta_val = config.beta
    
    # 1. 计算目标机构 0 的 VaR
    a_val_true = calculate_individual_var(config.target_i, alpha_val)
    print(f"机构 {config.target_i} 的 VaR_{alpha_val} (a_val) = {a_val_true:.4f}")
    
    # 2. 运行蒙特卡洛验证 (10万次模拟通常已具备很好的精度，如果您电脑配置高可以设为 100万)
    mc_covar = monte_carlo_covar(beta_val, a_val_true, num_simulations=200000)
    
    print("\n" + "="*50)
    print(f"=== Monte Carlo 验证结果 ===")
    print(f"目标机构 {config.target_i} (L_i) 损失: {a_val_true:.4f}")
    print(f"MC 模拟求得的系统 CoVaR_{beta_val}: {mc_covar:.4f}")
    print("="*50)
    
    # 结束计时
    end_time = time.time()
    
    # 计算执行时间
    elapsed_time = end_time - start_time
    print(f"代码执行时间: {elapsed_time:.4f} 秒")
    


    
    
    
    
    
    