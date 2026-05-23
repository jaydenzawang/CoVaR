#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:21:49 2026

@author: jaydenwang
"""


import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.stats import norm, chi2

import config
from copula import GaussianCopulaSingleFactor, TCopulaSingleFactor
from loss_model import PortfolioLoss
from spa import SaddlepointApproximation

from var_calc import calculate_individual_var

import time

# 初始化全局对象避免重复创建
copula = GaussianCopulaSingleFactor(config.rho, config.p)
# copula = TCopulaSingleFactor(config.rho, config.p, config.nu)
portfolio = PortfolioLoss(config.u, config.a, config.b)
spa = SaddlepointApproximation(portfolio, config.target_i)


def evaluate_conditional_cdf(l, a):
    """
    计算 F_{L|L_i=a}(l) 
    即: P(L <= l | L_i = a)
    """
    def integrand(Z):
        p_cond = copula.conditional_default_prob(Z)
        y = l - a 
        
        prob_SPA = spa.prob_L_minus_i(y, p_cond)
        p_cond_i = p_cond[config.target_i]
        pdf_Z = norm.pdf(Z)
        
        return prob_SPA * p_cond_i * pdf_Z

    # 由于正态分布尾部衰减快，在 [-6, 6] 积分已包含 99.999% 的概率质量
    integral_val, _ = quad(integrand, -6, 6, epsabs=1e-4)
    
    cdf_val = integral_val / config.p[config.target_i]
    return np.clip(cdf_val, 0.0, 1.0)


def calculate_covar(beta, a_val):
    """
    反解 CoVaR：找到使得 P(L <= l | L_i = a) = beta 的系统损失 l
    """
    print(f"开始求解 CoVaR (beta={beta}) 使得 P(L <= CoVaR | L_i={a_val:.2f}) = {beta}...")
    
    # 目标函数：CDF(l) - beta = 0
    def objective(l):
        cdf = evaluate_conditional_cdf(l, a_val)
        return cdf - beta
    
    # CoVaR 存在的范围：
    # 最小值：由于 L_{-i} >= 0，所以 L 至少为 a_val (此处略加epsilon防0)
    l_min = a_val + 1e-3 
    # 最大值：所有机构的最大总负债之和
    l_max = np.sum(config.u)
    
    # 检查边界，如果极端情况发生提早发现
    cdf_min = evaluate_conditional_cdf(l_min, a_val)
    cdf_max = evaluate_conditional_cdf(l_max, a_val)
    
    if cdf_max < beta:
        print("警告：即使所有机构违约，条件概率依然达不到 Beta，返回最大可能损失。")
        return l_max
        
    # 利用 Brentq 算法在这个区间内寻找使得目标函数为 0 的 l
    res = root_scalar(objective, bracket=[l_min, l_max], method='brentq', xtol=1e-2)
    return res.root

if __name__ == "__main__":
    
    # 开始计时
    start_time = time.time()
    print(f"当我们选择的copula是{copula}")
    
    # 1. 真实计算机构 0 的 VaR_{alpha} (alpha=0.95)
    a_val_true = calculate_individual_var(config.target_i, config.alpha)
    
    # # 信用风险中，若 0.95 置信度下 VaR 为 0，可能需要切换到 0.99 等更高置信度
    # # 否则 a=0 会导致后续计算条件概率退化
    # if a_val_true == 0.0:
    #     print("\n提示：因为 VaR_0.95 为 0，为了演示 CoVaR，我们将强制采用 alpha=0.99 来获取一个有意义的危机状态损失 a。")
    #     a_val_true = calculate_individual_var(config.target_i, 0.99)
        
    # 2. 用算出的 VaR 求解系统 CoVaR_beta
    
    covar_result = calculate_covar(config.beta, a_val_true)
    
    # CoVaR 的结果为：
    print(f"CoVaR(beta={config.beta},alpha={config.alpha},obligor={config.target_i})={covar_result:.4f}")
    
    # 结束计时
    end_time = time.time()
    
    # 计算执行时间
    elapsed_time = end_time - start_time
    print(f"代码执行时间: {elapsed_time:.4f} 秒")
    
    
    
    
    
    
    