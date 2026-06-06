#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:21:49 2026

@author: jaydenwang

功能：基于混合鞍点逼近 (Hybrid SPA) 计算等号危机条件下的系统共担风险值 CoVaR (L | L_i = a)
支持模型：一键切换 Gaussian / t / Skew-t Copula (m=1, xi=1)
"""

import numpy as np
import time

from scipy.integrate import quad, dblquad
from scipy.special import gamma, roots_genlaguerre
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import root_scalar
from scipy.stats import norm, chi2

import config
from copula import GaussianCopulaSingleFactor, TCopulaSingleFactor, SkewTCopulaSingleFactor
from loss_model import PortfolioLoss
from spa import SaddlepointApproximation
from var_calc import calculate_individual_var

np.random.seed(42)

# =============================================================================
# 【核心控制开关】 动态切换积分引擎
# =============================================================================
INTEGRATION_METHOD = 'fixed'  # 可选: 'fixed' (固定节点法) 或 'adaptive' (自适应高精度)

# 对象初始化
if config.COPULA_TYPE == 'Gaussian':
    copula = GaussianCopulaSingleFactor(config.rho, config.p)
elif config.COPULA_TYPE == 't':
    copula = TCopulaSingleFactor(config.rho, config.p, config.nu)
elif config.COPULA_TYPE == 'Skew-t':
    copula = SkewTCopulaSingleFactor(config.rho, config.p, config.nu, config.gamma)
else:
    raise ValueError(f"未知的 COPULA_TYPE: {config.COPULA_TYPE}")

portfolio = PortfolioLoss(config.u, config.a, config.b)
spa = SaddlepointApproximation(portfolio, config.target_i)


def evaluate_conditional_cdf_fixed(l, a):
    """引擎 A: 固定高斯求积 (极速版)"""
    N_GH, N_GL = 60, 60
    x_gh, w_gh = hermgauss(N_GH)
    Z_nodes = np.sqrt(2.0) * x_gh
    Z_weights = w_gh / np.sqrt(np.pi)

    alpha_laguerre = (config.nu / 2.0) - 1.0
    x_gl, w_gl = roots_genlaguerre(N_GL, alpha_laguerre)
    Lambda_nodes = 2.0 * x_gl
    Lambda_weights = w_gl / gamma(config.nu / 2.0)

    y = l - a
    is_default = (a > 0)
    uncond_prob = config.p[config.target_i] if is_default else (1.0 - config.p[config.target_i])
    integral_val = 0.0
    
    if isinstance(copula, GaussianCopulaSingleFactor):
        for i in range(N_GH):
            Z = Z_nodes[i]
            w_Z = Z_weights[i]
            p_cond = copula.conditional_default_prob(Z)
            prob_SPA = spa.prob_L_minus_i(y, p_cond)
            p_cond_i = p_cond[config.target_i] if is_default else (1.0 - p_cond[config.target_i])
            integral_val += w_Z * prob_SPA * p_cond_i
            
    elif isinstance(copula, (TCopulaSingleFactor, SkewTCopulaSingleFactor)):
        for j in range(N_GL):
            Lambda = Lambda_nodes[j]
            w_L = Lambda_weights[j]
            for i in range(N_GH):
                Z = Z_nodes[i]
                w_Z = Z_weights[i]
                if isinstance(copula, SkewTCopulaSingleFactor):
                    w_Z *= 2.0 * norm.cdf(copula.gamma * Z)
                p_cond = copula.conditional_default_prob(Z, Lambda)
                prob_SPA = spa.prob_L_minus_i(y, p_cond)
                p_cond_i = p_cond[config.target_i] if is_default else (1.0 - p_cond[config.target_i])
                integral_val += (w_Z * w_L) * prob_SPA * p_cond_i
                
    return np.clip(integral_val / uncond_prob, 0.0, 1.0)


def evaluate_conditional_cdf_adaptive(l, a):
    """引擎 B: 自适应数值积分 (高精度版)"""
    y = l - a
    is_default = (a > 0)
    uncond_prob = config.p[config.target_i] if is_default else (1.0 - config.p[config.target_i])
    
    if isinstance(copula, GaussianCopulaSingleFactor):
        # 1D 自适应积分
        def integrand_1d(Z):
            p_cond = copula.conditional_default_prob(Z)
            prob_SPA = spa.prob_L_minus_i(y, p_cond)
            p_cond_i = p_cond[config.target_i] if is_default else (1.0 - p_cond[config.target_i])
            return prob_SPA * p_cond_i * norm.pdf(Z)
            
        integral_val, _ = quad(integrand_1d, -np.inf, np.inf, epsabs=1e-5, epsrel=1e-5)
        
    elif isinstance(copula, (TCopulaSingleFactor, SkewTCopulaSingleFactor)):
        # 2D 自适应嵌套积分
        def integrand_2d(Z, Lambda):
            # 注: dblquad 传入参数的顺序为 (y, x) 即 (Z, Lambda)
            p_cond = copula.conditional_default_prob(Z, Lambda)
            prob_SPA = spa.prob_L_minus_i(y, p_cond)
            p_cond_i = p_cond[config.target_i] if is_default else (1.0 - p_cond[config.target_i])
            
            pdf_L = chi2.pdf(Lambda, config.nu)
            pdf_Z = norm.pdf(Z)
            if isinstance(copula, SkewTCopulaSingleFactor):
                pdf_Z *= 2.0 * norm.cdf(copula.gamma * Z)
                
            return prob_SPA * p_cond_i * pdf_Z * pdf_L
            
        # Lambda (x) 范围 [0, inf], Z (y) 范围 [-inf, inf]
        integral_val, _ = dblquad(
            integrand_2d, 0, np.inf, 
            lambda lam: -np.inf, lambda lam: np.inf, 
            epsabs=1e-5, epsrel=1e-5
        )
        
    return np.clip(integral_val / uncond_prob, 0.0, 1.0)


def calculate_covar(beta, a_val):
    """寻根算子调度器"""
    print(f"正在基于 [{INTEGRATION_METHOD}] 模式求解等式 CoVar...")
    
    def objective(l):
        if INTEGRATION_METHOD == 'adaptive':
            return evaluate_conditional_cdf_adaptive(l, a_val) - beta
        else:
            return evaluate_conditional_cdf_fixed(l, a_val) - beta
            
    l_min, l_max = a_val + 1e-3, np.sum(config.u)
    res = root_scalar(objective, bracket=[l_min, l_max], method='brentq', xtol=1e-2)
    return res.root


if __name__ == "__main__":
    start_time = time.time()
    print("=" * 60)
    print(f"模式: SPA + 等号场景 | 模型: {config.COPULA_TYPE} | 积分引擎: {INTEGRATION_METHOD}")
    print("=" * 60)
    
    a_val_true = calculate_individual_var(portfolio, copula, config.target_i, config.alpha)
    print(f"目标机构无条件 VaR = {a_val_true:.4f}\n")
        
    covar_result = calculate_covar(config.beta, a_val_true)
    
    print("-" * 60)
    print(f"CoVaR (Equality, {INTEGRATION_METHOD}) = {covar_result:.4f}")
    print(f"耗时: {time.time() - start_time:.4f} 秒")
    print("=" * 60)
    
    
    
    
    
    