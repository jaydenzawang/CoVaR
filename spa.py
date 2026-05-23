#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:21:38 2026

@author: jaydenwang
"""



import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import norm
import warnings
np.random.seed(42)

class SaddlepointApproximation:
    def __init__(self, portfolio, target_i):
        self.portfolio = portfolio
        self.target_i = target_i
        
        # 预先创建一个布尔掩码(mask)，用于在求和时排除目标机构 target_i
        self.mask = np.ones(len(self.portfolio.u), dtype=bool)
        self.mask[self.target_i] = False
        
    def cgf_deriv(self, s, p_cond, k):
        """利用向量化极速计算 CGF 的各阶导数"""
        # 1. 忽略 hyp1f1 可能产生的 overflow 警告 (我们在外层捕获处理)
        with np.errstate(over='ignore', invalid='ignore'):
            A = self.portfolio.A_j(s, p_cond)[self.mask]
            
            # 为计算 K0 (k=0) 单独预留
            if k == 0:
                return np.sum(np.log(A))
                
            A1 = self.portfolio.A_j_deriv(s, p_cond, 1)[self.mask]
            if k == 1:
                return np.sum(A1 / A)
                
            A2 = self.portfolio.A_j_deriv(s, p_cond, 2)[self.mask]
            if k == 2:
                return np.sum(A2 / A - (A1 / A)**2)
                
            A3 = self.portfolio.A_j_deriv(s, p_cond, 3)[self.mask]
            if k == 3:
                return np.sum(A3 / A - 3 * A2 * A1 / (A**2) + 2 * (A1 / A)**3)
                
            A4 = self.portfolio.A_j_deriv(s, p_cond, 4)[self.mask]
            if k == 4:
                return np.sum(A4 / A - 4 * A3 * A1 / (A**2) - 3 * (A2 / A)**2 + 12 * A2 * (A1**2) / (A**3) - 6 * (A1 / A)**4)

    def find_saddlepoint(self, y, p_cond):
        """求解鞍点方程找到 \hat{s}"""
        def obj(s):
            val = self.cgf_deriv(s, p_cond, 1) - y
            # 防止 hyp1f1 溢出导致 inf 或 nan
            if not np.isfinite(val):
                return 1e10 if s > 0 else -1e10
            return val
            
        try:
            # 优先在较小的区间内寻找 (避免 s 过大导致 e^(s*u) 溢出)
            res = root_scalar(obj, bracket=[-5, 5], method='brentq')
            return res.root
        except ValueError:
            try:
                # 如果找不到，略微放大区间。通常最大损失的上限不允许 s 跑到 20 以上
                res = root_scalar(obj, bracket=[-10, 10], method='brentq')
                return res.root
            except ValueError:
                return 0.0
    

    def prob_L_minus_i(self, y, p_cond):
        """SPA 计算尾部概率 P(L_{-i} <= y | \\Theta)"""
        # if y <= 0: return 0.0
        if y <= 0: 
            # 严格返回其他机构全不违约的联合概率
            return np.prod(1.0 - p_cond[self.mask])
            
        s_hat = self.find_saddlepoint(y, p_cond)
        
        # 处理中心极限定理退化情况
        if np.abs(s_hat) < 1e-5:
            mean = self.cgf_deriv(0, p_cond, 1)
            var = self.cgf_deriv(0, p_cond, 2)
            return norm.cdf((y - mean) / np.sqrt(var))
            
        # 向量化算出的各项 K
        K0 = self.cgf_deriv(s_hat, p_cond, 0)
        K2 = self.cgf_deriv(s_hat, p_cond, 2)
        K3 = self.cgf_deriv(s_hat, p_cond, 3)
        K4 = self.cgf_deriv(s_hat, p_cond, 4)
        
        # 鞍点公式计算
        w_hat = np.sign(s_hat) * np.sqrt(2 * max(s_hat * y - K0, 0)) 
        v_hat = s_hat * np.sqrt(K2)
        
        lambda_3 = K3 / (K2**(1.5))
        lambda_4 = K4 / (K2**2)
        
        term1 = norm.cdf(w_hat)
        
        # 忽略计算过程中产生的微小除 0 风险（SPA靠近均值时的奇点问题）
        with np.errstate(divide='ignore', invalid='ignore'):
            term2_calc = (1/w_hat - 1/v_hat - 1/(w_hat**3) + 1/(v_hat**3) + 
                          lambda_3/(2*v_hat**2) - (lambda_4/8 - 5*lambda_3**2 / 24)/v_hat)
            term2 = norm.pdf(w_hat) * term2_calc
            
        # 若奇点导致 nan，将其置零
        if np.isnan(term2):
            term2 = 0.0
            
        return np.clip(term1 + term2, 0.0, 1.0)
    
    
    
    
    
    
