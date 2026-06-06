#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:21:10 2026

@author: jaydenwang
"""

import numpy as np
from scipy.stats import norm, t
from scipy.special import roots_genlaguerre
from scipy.special import gamma as gamma_func
from scipy.optimize import root_scalar

np.random.seed(42)

class GaussianCopulaSingleFactor:
    """
    单因子高斯 Copula 依赖结构
    """
    def __init__(self, rho, p):
        self.rho = np.array(rho)
        self.p = np.array(p)
        # 违约阈值 x_i^d = \Phi^{-1}(1 - p_i)
        self.x_d = norm.ppf(1 - self.p)
        
    def conditional_default_prob(self, Z):
        """
        计算在给定单因子系统风险 Z 下的条件违约概率 p_{i|\Theta}
        """
        num = self.rho * Z - self.x_d
        den = np.sqrt(1 - self.rho**2)
        # 限制范围防止极值导致的浮点计算警告
        return norm.cdf(np.clip(num / den, -10, 10))
    
    
class TCopulaSingleFactor:
    """
    真正的单因子 t-Copula 依赖结构
    基于 Gaussian-Mixture 表达：引入共同冲击因子 Lambda ~ Chi^2(nu)
    """
    def __init__(self, rho, p, nu):
        self.rho = np.array(rho)
        self.p = np.array(p)
        self.nu = nu
        # 边缘分布是 t 分布，因此违约门槛 x_d 必须基于 t 分布的逆函数
        self.x_d = t.ppf(1 - self.p, df=self.nu)
        
    def conditional_default_prob(self, Z, Lambda):
        """
        计算给定 Z 和 Lambda 下的条件违约概率
        支持传入标量或一维批量数组 (用于 Monte Carlo)
        """
        Z = np.asarray(Z)
        Lambda = np.asarray(Lambda)
        
        # 提取共同冲击比例项 (对应公式中的分母 sqrt(Lambda / nu))
        # P(X_i > x_d | Z, Lambda) 等价于 P(eta_i > (x_d * sqrt(Lambda/nu) - rho*Z) / sqrt(1-rho^2))
        sqrt_term = np.sqrt(Lambda / self.nu)
        
        if Z.ndim > 0 and Lambda.ndim > 0:
            Z_term = np.outer(Z, self.rho)
            Lambda_term = np.outer(sqrt_term, self.x_d)
            num = Z_term - Lambda_term
        else:
            num = self.rho * Z - self.x_d * sqrt_term
            
        den = np.sqrt(1 - self.rho**2)
        
        # 给定 Z 和 Lambda 后，剩余的特质冲击 eta_i 服从标准正态分布
        # 所以根据对称性 P(eta_i > a) = Phi(-a)，映射函数必须使用 norm.cdf，绝不能用 t.cdf
        return norm.cdf(np.clip(num / den, -10, 10))

class SkewTCopulaSingleFactor:
    """
    单因子 Skew-t Copula 依赖结构 (m=1, xi=1)
    系统性风险因子 Z ~ Skew-Normal(0, 1, gamma), PDF: f_Z(z) = 2 * \phi(z) * \Phi(gamma * z)
    共同冲击因子 Lambda ~ Chi^2(nu)
    """
    def __init__(self, rho, p, nu, gamma_val):
        self.rho = np.array(rho)
        self.p = np.array(p)
        self.nu = nu
        self.gamma = gamma_val
        self.x_d = np.zeros_like(self.p)
        
        # 1. 预配置全概率数值积分的节点与权重，用于快速反解阈值 x_d
        N_GH, N_GL = 60, 60
        
        # Z 的高斯-赫米特求积调整 (融入偏正态分布的 2 * \Phi(gamma * z))
        nodes_GH, weights_GH = np.polynomial.hermite.hermgauss(N_GH)
        Z_nodes = np.sqrt(2.0) * nodes_GH
        Z_weights = (weights_GH / np.sqrt(np.pi)) * 2.0 * norm.cdf(self.gamma * Z_nodes)
        
        # Lambda 的广义高斯-拉盖尔求积调整
        nodes_GL, weights_GL = roots_genlaguerre(N_GL, alpha=self.nu / 2.0 - 1.0)
        Lambda_nodes = 2.0 * nodes_GL
        Lambda_weights = weights_GL / gamma_func(self.nu / 2.0)
        
        # 2. 逐个机构通过寻根法反解违约门槛 x_i^d，使得无条件违约概率等于 p_i
        # print("正在构建 Skew-t Copula 违约阈值矩阵 (数值反解中)...")
        for i in range(len(self.p)):
            rho_i = self.rho[i]
            p_i = self.p[i]
            
            def unconditional_prob(x):
                # 建立二维因子网格
                Z_grid = Z_nodes[:, None]
                L_grid = Lambda_nodes[None, :]
                # 计算条件违约概率
                num = rho_i * Z_grid - x * np.sqrt(L_grid / self.nu)
                den = np.sqrt(1 - rho_i**2)
                p_cond = norm.cdf(np.clip(num / den, -10, 10))
                # 二维积分求和
                return np.sum(p_cond * Z_weights[:, None] * Lambda_weights[None, :])
            
            # 使用 Brentq 算法寻根
            res = root_scalar(lambda x: unconditional_prob(x) - p_i, bracket=[-10, 10], method='brentq')
            self.x_d[i] = res.root

    def conditional_default_prob(self, Z, Lambda):
        """
        计算给定 Z 和 Lambda 条件下的违约概率
        由于系统因子和冲击因子确定后，特有风险 \eta_i 仍为标准正态分布，
        因此条件概率的函数形式与标准 t-Copula 保持一致。
        """
        Z = np.asarray(Z)
        Lambda = np.asarray(Lambda)
        num = self.rho * Z - self.x_d * np.sqrt(Lambda / self.nu)
        den = np.sqrt(1 - self.rho**2)
        return norm.cdf(np.clip(num / den, -10, 10))
