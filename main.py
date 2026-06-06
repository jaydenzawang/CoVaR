#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:21:49 2026

@author: jaydenwang

功能：基于混合鞍点逼近 (Hybrid SPA) 计算等号危机条件下的系统共担风险值 CoVaR (L | L_i = a)
支持模型：一键切换 Gaussian / t / Skew-t Copula 并支持任意参数 xi 的 Gumbel 因子联结
"""

import numpy as np
import time

from scipy.integrate import quad, dblquad
from scipy.special import gamma, roots_genlaguerre
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import root_scalar
from scipy.stats import norm, chi2, skewnorm

import config
from copula import GaussianCopulaSingleFactor, TCopulaSingleFactor, SkewTCopulaSingleFactor
from loss_model import PortfolioLoss
from spa import SaddlepointApproximation
from var_calc import calculate_individual_var
from gumbel_interplay import gumbel_copula_pdf

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
    """引擎 A: 固定高斯求积网络（已融合同步广播的 Gumbel 依赖测度）"""
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
    
    xi_val = config.XI if hasattr(config, 'XI') else 1.0
    
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
            v_val = chi2.sf(Lambda, config.nu)  # 冲击因子的生存函数轴
            
            for i in range(N_GH):
                Z = Z_nodes[i]
                w_Z = Z_weights[i]
                
                # 依据模型抓取 Z 轴的边际累积概率测度 u
                if isinstance(copula, SkewTCopulaSingleFactor):
                    w_Z *= 2.0 * norm.cdf(copula.gamma * Z)
                    u_val = skewnorm.cdf(Z, copula.gamma)
                else:
                    u_val = norm.cdf(Z)
                
                # 计算当前节点处的 Gumbel 相互作用密度校正因子
                gumbel_factor = gumbel_copula_pdf(u_val, v_val, xi_val)
                
                p_cond = copula.conditional_default_prob(Z, Lambda)
                prob_SPA = spa.prob_L_minus_i(y, p_cond)
                p_cond_i = p_cond[config.target_i] if is_default else (1.0 - p_cond[config.target_i])
                
                # 测度变换：原权重乘上 Gumbel 乘子
                integral_val += (w_Z * w_L) * gumbel_factor * prob_SPA * p_cond_i
                
    return np.clip(integral_val / uncond_prob, 0.0, 1.0)


def evaluate_conditional_cdf_adaptive(l, a):
    """引擎 B: 自适应连续数值积分（已在被积核心动态注入 Gumbel 测度）"""
    y = l - a
    is_default = (a > 0)
    uncond_prob = config.p[config.target_i] if is_default else (1.0 - config.p[config.target_i])
    
    xi_val = config.XI if hasattr(config, 'XI') else 1.0
    
    if isinstance(copula, GaussianCopulaSingleFactor):
        def integrand_1d(Z):
            p_cond = copula.conditional_default_prob(Z)
            prob_SPA = spa.prob_L_minus_i(y, p_cond)
            p_cond_i = p_cond[config.target_i] if is_default else (1.0 - p_cond[config.target_i])
            return prob_SPA * p_cond_i * norm.pdf(Z)
            
        integral_val, _ = quad(integrand_1d, -np.inf, np.inf, epsabs=1e-5, epsrel=1e-5)
        
    elif isinstance(copula, (TCopulaSingleFactor, SkewTCopulaSingleFactor)):
        def integrand_2d(Z, Lambda):
            p_cond = copula.conditional_default_prob(Z, Lambda)
            prob_SPA = spa.prob_L_minus_i(y, p_cond)
            p_cond_i = p_cond[config.target_i] if is_default else (1.0 - p_cond[config.target_i])
            
            pdf_L = chi2.pdf(Lambda, config.nu)
            pdf_Z = norm.pdf(Z)
            
            if isinstance(copula, SkewTCopulaSingleFactor):
                pdf_Z *= 2.0 * norm.cdf(copula.gamma * Z)
                u_val = skewnorm.cdf(Z, copula.gamma)
            else:
                u_val = norm.cdf(Z)
                
            v_val = chi2.sf(Lambda, config.nu)
            # 计算当前因子点上的 Gumbel 联结系数
            gumbel_factor = gumbel_copula_pdf(u_val, v_val, xi_val)
                
            return prob_SPA * p_cond_i * pdf_Z * pdf_L * gumbel_factor
            
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
    print(f"模式: SPA + 等号场景 | 模型: {config.COPULA_TYPE} | 依赖系数 XI: {config.XI} | 积分引擎: {INTEGRATION_METHOD}")
    print("=" * 60)
    
    a_val_true = calculate_individual_var(portfolio, copula, config.target_i, config.alpha)
    print(f"目标机构无条件 VaR = {a_val_true:.4f}\n")
        
    covar_result = calculate_covar(config.beta, a_val_true)
    
    print("-" * 60)
    print(f"CoVaR (Equality, {INTEGRATION_METHOD}) = {covar_result:.4f}")
    print(f"耗时: {time.time() - start_time:.4f} 秒")
    print("=" * 60)