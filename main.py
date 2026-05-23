#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:21:49 2026

@author: jaydenwang
"""


import numpy as np
from scipy.integrate import quad, dblquad
from scipy.optimize import root_scalar
from scipy.stats import norm, chi2

from scipy.special import gamma, roots_genlaguerre
from numpy.polynomial.hermite import hermgauss

import config
from copula import GaussianCopulaSingleFactor, TCopulaSingleFactor
from loss_model import PortfolioLoss
from spa import SaddlepointApproximation

from var_calc import calculate_individual_var

import time

np.random.seed(42)

# 初始化全局对象避免重复创建
copula = GaussianCopulaSingleFactor(config.rho, config.p)
# copula = TCopulaSingleFactor(config.rho, config.p, config.nu)
portfolio = PortfolioLoss(config.u, config.a, config.b)
spa = SaddlepointApproximation(portfolio, config.target_i)


# Fixed quadrature
def evaluate_conditional_cdf(l, a):
    """
    计算 F_{L|L_i=a}(l) 
    完全采用极速高斯求积 (Gaussian Quadrature) 替代 scipy.integrate 的自适应积分
    """
    
    # 全局预计算求积节点 (Quadrature Nodes & Weights)
    # N=15 到 20 通常已具备极高的精度
    # ==========================================
    N_GH = 100 # Gauss-Hermite 节点数
    N_GL = 10 # Gauss-Laguerre 节点数

    # 1. 预计算 Z 的 Gauss-Hermite 节点和权重
    x_gh, w_gh = hermgauss(N_GH)
    Z_nodes = np.sqrt(2) * x_gh
    Z_weights = w_gh / np.sqrt(np.pi)

    # 2. 预计算 Lambda 的 Generalized Gauss-Laguerre 节点和权重
    # 注意：这依赖于你 config 中配置的 t-copula 自由度 nu
    alpha_laguerre = (config.nu / 2.0) - 1.0
    x_gl, w_gl = roots_genlaguerre(N_GL, alpha_laguerre)
    Lambda_nodes = 2.0 * x_gl
    Lambda_weights = w_gl / gamma(config.nu / 2.0)


    y = l - a
    is_default = (a > 0)
    uncond_prob = config.p[config.target_i] if is_default else (1.0 - config.p[config.target_i])
    
    integral_val = 0.0
    
    # 场景 1: Gaussian Copula (1D Gauss-Hermite)
    if isinstance(copula, GaussianCopulaSingleFactor):
        # 仅对 Z 循环 N_GH 次 (如 15 次)
        for i in range(N_GH):
            Z = Z_nodes[i]
            w_Z = Z_weights[i]
            
            p_cond = copula.conditional_default_prob(Z)
            prob_SPA = spa.prob_L_minus_i(y, p_cond)
            p_cond_i = p_cond[config.target_i] if is_default else (1.0 - p_cond[config.target_i])
            
            integral_val += w_Z * prob_SPA * p_cond_i
            
    # 场景 2: t-Copula (2D Gauss-Hermite + Gauss-Laguerre)
    elif isinstance(copula, TCopulaSingleFactor):
        # 嵌套循环: N_GL * N_GH 次 (如 15 * 15 = 225 次)
        for j in range(N_GL):
            Lambda = Lambda_nodes[j]
            w_L = Lambda_weights[j]
            
            for i in range(N_GH):
                Z = Z_nodes[i]
                w_Z = Z_weights[i]
                
                p_cond = copula.conditional_default_prob(Z, Lambda)
                prob_SPA = spa.prob_L_minus_i(y, p_cond)
                p_cond_i = p_cond[config.target_i] if is_default else (1.0 - p_cond[config.target_i])
                
                # 联合权重相乘
                integral_val += (w_Z * w_L) * prob_SPA * p_cond_i
    
    else:
        raise ValueError("Unsupported Copula type.")

    # 计算条件概率并强制界定
    cdf_val = integral_val / uncond_prob
    return np.clip(cdf_val, 0.0, 1.0)


# # Adaptive integration: dblquad
# def evaluate_conditional_cdf(l, a):
#     """
#     计算 F_{L|L_i=a}(l) 
#     即: P(L <= l | L_i = a)
#     包含高斯 Copula 的 1D 积分与 t-Copula 的 2D 积分动态适配
#     """
#     y = l - a
#     # 判断是否为违约状态，提取对应的无条件概率
#     is_default = (a > 0)
#     uncond_prob = config.p[config.target_i] if is_default else (1.0 - config.p[config.target_i])
    
#     # 场景 1: Gaussian Copula (单因子 Z 积分)
#     if isinstance(copula, GaussianCopulaSingleFactor):
#         def integrand_gauss(Z):
#             p_cond = copula.conditional_default_prob(Z)
#             prob_SPA = spa.prob_L_minus_i(y, p_cond)
#             p_cond_i = p_cond[config.target_i] if is_default else (1.0 - p_cond[config.target_i])
            
#             return prob_SPA * p_cond_i * norm.pdf(Z)
            
#         integral_val, _ = quad(integrand_gauss, -6, 6, epsabs=1e-4)
        
#     # 场景 2: t-Copula (Z 与 共同冲击因子 Lambda 的联合二维积分)
#     elif isinstance(copula, TCopulaSingleFactor):
#         def integrand_t(Lambda, Z):
#             # 注意 dblquad 传参顺序：先传内层积分变量 Lambda，再传外层积分变量 Z
#             p_cond = copula.conditional_default_prob(Z, Lambda)
#             prob_SPA = spa.prob_L_minus_i(y, p_cond)
#             p_cond_i = p_cond[config.target_i] if is_default else (1.0 - p_cond[config.target_i])
            
#             pdf_Z = norm.pdf(Z)
#             pdf_Lambda = chi2.pdf(Lambda, df=config.nu)
#             return prob_SPA * p_cond_i * pdf_Z * pdf_Lambda
            
#         # Z 积分区间: [-6, 6]
#         # Lambda 积分区间: [1e-4, np.inf] (Chi^2(nu) 分布支撑集，加入 1e-4 截断防止除0)
#         integral_val, _ = dblquad(
#             integrand_t, 
#             -6.0, 6.0,                  # 外层 Z 的积分上下限
#             lambda Z: 1e-4,             # 内层 Lambda 的积分下限
#             lambda Z: np.inf,           # 内层 Lambda 的积分上限
#             epsabs=1e-4
#         )
    
#     else:
#         raise ValueError("Unsupported Copula type.")

#     # 计算条件概率并强制界定在 [0, 1] 之间
#     cdf_val = integral_val / uncond_prob
#     return np.clip(cdf_val, 0.0, 1.0)


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
    
    # 1. 真实计算机构 0 的 VaR_{alpha} (alpha=0.99 or 0.999)
    a_val_true = calculate_individual_var(portfolio, copula, config.target_i, config.alpha)
    print(f"机构 {config.target_i} 的无条件 VaR_{config.alpha} = {a_val_true:.4f}")
    
    # # 可选：校验当前选择的 Copula 边缘分布一致性
    # from var_calc import verify_copula_marginal
    # p_check = verify_copula_marginal(copula, config.target_i)
    # print(f"当前 Copula 积分导出的无条件违约概率校验值: {p_check:.4f} (原始配置值: {config.p[config.target_i]:.4f})")
        
      
    # 2. 用算出的 VaR 求解系统 CoVaR_beta
    
    covar_result = calculate_covar(config.beta, a_val_true)
    
    # CoVaR 的结果为：
    print(f"CoVaR(beta={config.beta},alpha={config.alpha},obligor={config.target_i})={covar_result:.4f}")
    
    # 结束计时
    end_time = time.time()
    
    # 计算执行时间
    elapsed_time = end_time - start_time
    print(f"代码执行时间: {elapsed_time:.4f} 秒")
    
    
    
    
    
    
    