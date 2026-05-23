#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:21:10 2026

@author: jaydenwang
"""

import numpy as np
from scipy.stats import norm, t

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
        
        # 【核心修正】：给定 Z 和 Lambda 后，剩余的特质冲击 eta_i 服从标准正态分布！
        # 所以根据对称性 P(eta_i > a) = Phi(-a)，映射函数必须使用 norm.cdf，绝不能用 t.cdf
        return norm.cdf(np.clip(num / den, -10, 10))