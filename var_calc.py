#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:33:54 2026

@author: jaydenwang

var_calc.py

功能：
1. 精准解析计算单个债务人/金融机构的无条件个体 VaR_alpha。
2. 通过全概率数值积分校验各类 Copula (Gaussian / t / Skew-t) 在 Gumbel 因子耦合（XI >= 1）下的边缘分布一致性。
"""

import numpy as np
from scipy.stats import beta as scipy_beta
from scipy.stats import norm, chi2, skewnorm
from scipy.integrate import quad, dblquad

# 导入配置和 Gumbel 联结函数
import config
from gumbel_interplay import gumbel_copula_pdf
# 导入所有的 Copula 结构，确保类型识别完整
from copula import GaussianCopulaSingleFactor, TCopulaSingleFactor, SkewTCopulaSingleFactor

np.random.seed(42)

def calculate_individual_var(portfolio, copula, idx, alpha):
    """
    计算单个金融机构的无条件个体 VaR_alpha (解析解版本)
    由于个体分布属于离散跳跃与连续 Beta 的混合分布，此解析算法具备 100% 的绝对精度。
    注：即便 XI > 1，由于阈值 x_d 已完成一致性校准，该边际解析解依然完全适用。
    """
    u_i = portfolio.u[idx]
    a_i = portfolio.a[idx]
    b_i = portfolio.b[idx]
    
    if hasattr(copula, 'p'):
        p_i = copula.p[idx]
    else:
        raise ValueError("传入的 Copula 对象未配置无条件违约概率矩阵 'p'。")
        
    prob_no_default = 1.0 - p_i
    
    # 边界保护机制：如果置信度 alpha 低于不违约概率，其 VaR 严格为 0
    if alpha <= prob_no_default:
        return 0.0
        
    # 映射到违约条件下的截断 Beta 分布百分位数
    target_prob = (alpha - prob_no_default) / p_i
    epsilon_quantile = scipy_beta.ppf(target_prob, a_i, b_i)
    
    return u_i * epsilon_quantile


def verify_copula_marginal(copula, idx):
    """
    【学术与数学严谨性双重强化验证】
    通过对系统性风险因子空间进行全概率自适应积分，计算集成的边缘违约概率。
    已全面升级：支持 XI > 1 时 Gumbel Copula 带来的 Z 和 Lambda 联合密度变形校验。
    """
    # 获取全局配置中的依赖参数 xi
    xi_val = config.XI if hasattr(config, 'XI') else 1.0

    # 场景 A: 单因子高斯 Copula (1D 因子空间积分，天然无 Lambda，不受 XI 影响)
    if isinstance(copula, GaussianCopulaSingleFactor):
        def integrand_gauss(Z):
            return copula.conditional_default_prob(Z)[idx] * norm.pdf(Z)
        
        integrated_p, _ = quad(integrand_gauss, -np.inf, np.inf)
        return integrated_p
        
    # 场景 B: 单因子 t-Copula (2D 因子空间积分，融入 Gumbel 联合依赖测度)
    elif isinstance(copula, TCopulaSingleFactor):
        def integrand_t(Z, Lambda):
            p_cond = copula.conditional_default_prob(Z, Lambda)[idx]
            pdf_L = chi2.pdf(Lambda, copula.nu)
            pdf_Z = norm.pdf(Z)
            
            # 【核心修改点】计算当前位置的 Gumbel 联结概率密度修正项
            u_val = norm.cdf(Z)
            v_val = chi2.sf(Lambda, copula.nu)
            gumbel_factor = gumbel_copula_pdf(u_val, v_val, xi_val)
            
            # 将依赖修正项乘入联合分布测度
            return p_cond * pdf_Z * pdf_L * gumbel_factor
            
        integrated_p, _ = dblquad(
            integrand_t, 0, np.inf,
            lambda lam: -np.inf, lambda lam: np.inf
        )
        return integrated_p

    # 场景 C: 单因子 Skew-t Copula (2D 因子空间积分，融入偏正态分布与 Gumbel 联合依赖测度)
    elif isinstance(copula, SkewTCopulaSingleFactor):
        def integrand_skew_t(Z, Lambda):
            p_cond = copula.conditional_default_prob(Z, Lambda)[idx]
            pdf_L = chi2.pdf(Lambda, copula.nu)
            # 引入单因子 Skew-Normal 边际概率密度：2 * \phi(z) * \Phi(\gamma * z)
            pdf_Z = norm.pdf(Z) * 2.0 * norm.cdf(copula.gamma * Z)
            
            # 【核心修改点】计算当前位置的 Gumbel 联结概率密度修正项
            # 注意：Z 轴必须采用一维偏正态分布的累积分布函数 skewnorm.cdf
            u_val = skewnorm.cdf(Z, copula.gamma)
            v_val = chi2.sf(Lambda, copula.nu)
            gumbel_factor = gumbel_copula_pdf(u_val, v_val, xi_val)
            
            # 将依赖修正项乘入联合分布测度
            return p_cond * pdf_Z * pdf_L * gumbel_factor
            
        integrated_p, _ = dblquad(
            integrand_skew_t, 0, np.inf,
            lambda lam: -np.inf, lambda lam: np.inf
        )
        return integrated_p
        
    else:
        raise TypeError("未知的 Copula 依赖类型，无法进行全概率边缘校验。")