#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:33:54 2026

@author: jaydenwang
"""

import numpy as np
from scipy.stats import beta
import config

np.random.seed(42)

def calculate_individual_var(target_i, alpha):
    """
    计算单个金融机构的无条件 VaR_alpha
    
    参数:
    target_i: 机构的索引 (int)
    alpha: 置信水平 (float)
    """
    # 获取该机构的参数
    u_i = config.u[target_i]
    p_i = config.p[target_i]
    a_i = config.a[target_i]
    b_i = config.b[target_i]
    
    # 不违约的概率
    prob_no_default = 1.0 - p_i
    
    print(f"机构 {target_i} 的无条件违约概率为: {p_i:.4f} (不违约概率: {prob_no_default:.4f})")
    
    # 如果要求的置信水平 alpha 包含在“不违约”的概率质量中，则 VaR 为 0
    if alpha <= prob_no_default:
        print(f"警告: 置信水平 alpha ({alpha}) <= 不违约概率 ({prob_no_default})。")
        print("在这样的置信水平下，预期该机构不会发生违约，因此 VaR 严格为 0。")
        return 0.0
        
    # 否则，需要计算违约条件下的 LGD 分位数
    # P(\epsilon_i <= x / u_i) = target_prob
    target_prob = (alpha - prob_no_default) / p_i
    
    # 使用 Beta 分布的 PPF (Percent Point Function，即 CDF 的逆函数) 求解分位数
    epsilon_quantile = beta.ppf(target_prob, a_i, b_i)
    
    # 最终的 VaR = 总负债 * 违约损失率分位数
    var_alpha = u_i * epsilon_quantile
    
    return var_alpha

if __name__ == "__main__":
    # 使用 config 中定义的 alpha (0.95) 和 target_i (0)
    alpha_val = config.alpha
    target_idx = config.target_i
    
    var_result = calculate_individual_var(target_idx, alpha_val)
    
    print("-" * 50)
    print(f"机构 {target_idx} 总负债 (u_i) = {config.u[target_idx]:.2f}")
    print(f"计算所得 VaR_{alpha_val} = {var_result:.4f}")
    print("-" * 50)
    



    
    
    