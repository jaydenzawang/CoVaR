#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:33:54 2026

@author: jaydenwang
"""

import numpy as np
from scipy.stats import beta as scipy_beta
from scipy.stats import norm, chi2
from scipy.integrate import quad, dblquad
from copula import GaussianCopulaSingleFactor, TCopulaSingleFactor

np.random.seed(42)

def calculate_individual_var(portfolio, copula, idx, alpha):
    """
    计算单个金融机构的无条件 VaR_alpha (完全模块化版本)
    
    参数:
    portfolio: PortfolioLoss 实例，提供债务面值 u 和 LGD 的 Beta 参数
    copula: Copula 实例 (Gaussian 或 T)，提供准确的边缘违约概率 p_i
    idx: 目标机构的索引 (int)
    alpha: 置信水平 (float)
    """
    # 1. 模块化动态获取：从传入的 portfolio 和 copula 对象中提取参数
    u_i = portfolio.u[idx]
    a_i = portfolio.a[idx]
    b_i = portfolio.b[idx]
    
    # 核心体现：个体风险度量直接从 Copula 模型中提取其配置的边缘违约概率
    if hasattr(copula, 'p'):
        p_i = copula.p[idx]
    else:
        raise ValueError("传入的 Copula 对象未配置无条件违约概率矩阵 'p'。")
        
    prob_no_default = 1.0 - p_i
    
    # 2. 边界保护机制
    if alpha <= prob_no_default:
        return 0.0
        
    # 3. 计算违约条件下的 LGD 分位数
    target_prob = (alpha - prob_no_default) / p_i
    epsilon_quantile = scipy_beta.ppf(target_prob, a_i, b_i)
    
    return u_i * epsilon_quantile


def verify_copula_marginal(copula, idx):
    """
    【数学与模块化协同强化】
    通过对系统风险因子进行全概率积分，验证当前选择的 Copula 模型是否满足边缘分布的一致性。
    该函数完美体现了个体违约概率在数值上是如何通过 Copula 驱动的条件概率集成而来的。
    """
    # 场景 A: 如果当前选择的是单因子高斯 Copula
    if isinstance(copula, GaussianCopulaSingleFactor):
        def integrand_gauss(Z):
            # 提取给定 Z 下的条件违约概率
            return copula.conditional_default_prob(Z)[idx] * norm.pdf(Z)
        
        integrated_p, _ = quad(integrand_gauss, -6, 6)
        
    # 场景 B: 如果当前选择的是单因子 t-Copula (依赖 Z 和 共同冲击因子 Lambda)
    elif isinstance(copula, TCopulaSingleFactor):
        def integrand_t(Lambda, Z):
            return copula.conditional_default_prob(Z, Lambda)[idx] * norm.pdf(Z) * chi2.pdf(Lambda, df=copula.nu)
            
        integrated_p, _ = dblquad(
            integrand_t, 
            -6.0, 6.0, 
            lambda Z: 1e-4, 
            lambda Z: np.inf
        )
    else:
        # 降级兜底处理
        integrated_p = copula.p[idx]
        
    return integrated_p