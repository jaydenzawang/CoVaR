#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 21:43:43 2026

@author: jaydenwang

功能：基于混合鞍点逼近 (Hybrid SPA) 计算不等号危机条件下的系统共担风险值 CoVaR (L | L_i >= a)
支持：一键切换 Gaussian / t / Skew-t Copula 
控制：支持 'fixed' 固定求积 与 'adaptive' 自适应积分双引擎
"""

import numpy as np
import time
from scipy.optimize import root_scalar
from scipy.stats import norm, chi2, beta as scipy_beta
from scipy.special import gamma, roots_genlaguerre
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.integrate import dblquad, tplquad  # 导入高维自适应积分器

import config
from copula import GaussianCopulaSingleFactor, TCopulaSingleFactor, SkewTCopulaSingleFactor
from loss_model import PortfolioLoss
from spa import SaddlepointApproximation
from var_calc import calculate_individual_var

# =============================================================================
# 【核心控制开关】 动态切换积分引擎
# =============================================================================
INTEGRATION_METHOD = 'adaptive'  # 可选: 'fixed' (固定节点版) 或 'adaptive' (高精度自适应版)

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


def evaluate_conditional_cdf_ineq_fixed(l, a):
    """引擎 A: 固定多层高斯求积网络"""
    target_i = config.target_i
    u_i = config.u[target_i]
    a_i, b_i = portfolio.a[target_i], portfolio.b[target_i]
    A_bound, B_bound = a / u_i, 1.0
    
    N_Leg = 32
    x_leg, w_leg = leggauss(N_Leg)
    y_nodes = 0.5 * (B_bound - A_bound) * x_leg + 0.5 * (B_bound + A_bound)
    y_weights = 0.5 * (B_bound - A_bound) * w_leg
    pdf_vals = scipy_beta.pdf(y_nodes, a_i, b_i)

    N_GH, N_GL = 40, 40
    x_gh, w_gh = hermgauss(N_GH)
    Z_nodes = np.sqrt(2.0) * x_gh
    Z_weights = w_gh / np.sqrt(np.pi)

    alpha_laguerre = (config.nu / 2.0) - 1.0
    x_gl, w_gl = roots_genlaguerre(N_GL, alpha_laguerre)
    Lambda_nodes = 2.0 * x_gl
    Lambda_weights = w_gl / gamma(config.nu / 2.0)

    denom = (1.0 - scipy_beta.cdf(A_bound, a_i, b_i)) * config.p[target_i]
    integral_val = 0.0

    if isinstance(copula, GaussianCopulaSingleFactor):
        for i in range(N_GH):
            Z = Z_nodes[i]
            w_Z = Z_weights[i]
            p_cond = copula.conditional_default_prob(Z)
            p_cond_i = p_cond[target_i]
            
            inner_integral = 0.0
            for k in range(N_Leg):
                rem_l = l - u_i * y_nodes[k]
                inner_integral += y_weights[k] * spa.prob_L_minus_i(rem_l, p_cond) * pdf_vals[k]
            integral_val += w_Z * inner_integral * p_cond_i

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
                p_cond_i = p_cond[target_i]
                
                inner_integral = 0.0
                for k in range(N_Leg):
                    rem_l = l - u_i * y_nodes[k]
                    inner_integral += y_weights[k] * spa.prob_L_minus_i(rem_l, p_cond) * pdf_vals[k]
                integral_val += (w_Z * w_L) * inner_integral * p_cond_i

    return np.clip(integral_val / denom, 0.0, 1.0)


def evaluate_conditional_cdf_ineq_adaptive(l, a):
    """引擎 B: 自适应高维连续积分 (学术校验版)"""
    target_i = config.target_i
    u_i = config.u[target_i]
    a_i, b_i = portfolio.a[target_i], portfolio.b[target_i]
    A_bound = a / u_i
    
    denom = (1.0 - scipy_beta.cdf(A_bound, a_i, b_i)) * config.p[target_i]

    if isinstance(copula, GaussianCopulaSingleFactor):
        # 2D 自适应积分: (eps, Z)
        def integrand_2d(eps, Z):
            # dblquad 传参顺序为 (y, x) -> y=eps, x=Z
            p_cond = copula.conditional_default_prob(Z)
            p_cond_i = p_cond[target_i]
            rem_l = l - u_i * eps
            prob_SPA = spa.prob_L_minus_i(rem_l, p_cond)
            pdf_eps = scipy_beta.pdf(eps, a_i, b_i)
            return prob_SPA * pdf_eps * p_cond_i * norm.pdf(Z)
            
        # Z 为外层 x 轴 [-inf, inf]，eps 为内层 y 轴 [A_bound, 1.0]
        integral_val, _ = dblquad(
            integrand_2d, -np.inf, np.inf, 
            lambda z: A_bound, lambda z: 1.0, 
            epsabs=1e-4, epsrel=1e-4
        )

    elif isinstance(copula, (TCopulaSingleFactor, SkewTCopulaSingleFactor)):
        # 3D 嵌套自适应积分: (eps, Z, Lambda)
        def integrand_3d(eps, Z, Lambda):
            # tplquad 传参顺序为 (z, y, x) -> z=eps, y=Z, x=Lambda
            p_cond = copula.conditional_default_prob(Z, Lambda)
            p_cond_i = p_cond[target_i]
            rem_l = l - u_i * eps
            prob_SPA = spa.prob_L_minus_i(rem_l, p_cond)
            
            pdf_eps = scipy_beta.pdf(eps, a_i, b_i)
            pdf_L = chi2.pdf(Lambda, config.nu)
            pdf_Z = norm.pdf(Z)
            if isinstance(copula, SkewTCopulaSingleFactor):
                pdf_Z *= 2.0 * norm.cdf(copula.gamma * Z)
                
            return prob_SPA * pdf_eps * p_cond_i * pdf_Z * pdf_L
            
        # Lambda 为 x 轴 [0, inf], Z 为 y 轴 [-inf, inf], eps 为 z 轴 [A_bound, 1.0]
        integral_val, _ = tplquad(
            integrand_3d, 0, np.inf,
            lambda lam: -np.inf, lambda lam: np.inf,
            lambda lam, z_val: A_bound, lambda lam, z_val: 1.0,
            epsabs=1e-4, epsrel=1e-4
        )

    return np.clip(integral_val / denom, 0.0, 1.0)


def calculate_covar_ineq(beta, a_val):
    """寻根算子调度器"""
    print(f"正在基于 [{INTEGRATION_METHOD}] 模式求解不等式 CoVar...")
    
    def objective(l):
        if INTEGRATION_METHOD == 'adaptive':
            return evaluate_conditional_cdf_ineq_adaptive(l, a_val) - beta
        else:
            return evaluate_conditional_cdf_ineq_fixed(l, a_val) - beta
        
    l_min, l_max = a_val + 1e-3, np.sum(config.u)
    res = root_scalar(objective, bracket=[l_min, l_max], method='brentq', xtol=1e-2)
    return res.root


if __name__ == "__main__":
    start_time = time.time()
    print("=" * 60)
    print(f"模式: SPA + 不等号场景 | 模型: {config.COPULA_TYPE} | 积分引擎: {INTEGRATION_METHOD}")
    print("=" * 60)
    
    a_val_true = calculate_individual_var(portfolio, copula, config.target_i, config.alpha)
    print(f"目标机构无条件 VaR = {a_val_true:.4f}\n")
    
    covar_result = calculate_covar_ineq(config.beta, a_val_true)
    
    print("-" * 60)
    print(f"CoVaR (Inequality, {INTEGRATION_METHOD}) = {covar_result:.4f}")
    print(f"总计算耗时: {time.time() - start_time:.4f} 秒")
    print("=" * 60)