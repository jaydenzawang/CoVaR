#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:21:23 2026

@author: jaydenwang
"""


import numpy as np
from scipy.special import hyp1f1, poch
from scipy.stats import beta as scipy_beta
np.random.seed(42)

class PortfolioLoss:
    """
    投资组合损失模型，包含 LGD 的 Beta 分布 MGF 计算 (向量化版本)
    """
    def __init__(self, u, a, b):
        self.u = np.array(u)
        self.a = np.array(a)
        self.b = np.array(b)
    
    # 用于SPA
    def mgf_epsilon(self, s):
        """计算 Beta 分布的矩母函数 (同时计算所有机构)"""
        # s是标量，self.u是数组，hyp1f1可以直接处理数组输入
        return hyp1f1(self.a, self.a + self.b, s * self.u)
        
    def mgf_epsilon_deriv(self, s, k=1):
        """计算 Beta 分布 MGF 的第 k 阶导数 (同时计算所有机构)"""
        coef = poch(self.a, k) / poch(self.a + self.b, k)
        return coef * hyp1f1(self.a + k, self.a + self.b + k, s * self.u)

    def A_j(self, s, p_cond):
        """计算 A_j(s|\\theta) (返回维度为 N 的数组)"""
        return 1 - p_cond + p_cond * self.mgf_epsilon(s)
        
    def A_j_deriv(self, s, p_cond, k):
        """计算 A_j 的 k 阶导数 (返回维度为 N 的数组)"""
        return p_cond * (self.u**k) * self.mgf_epsilon_deriv(s, k)
    


