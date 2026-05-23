#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:21:23 2026

@author: jaydenwang
"""


import numpy as np
from scipy.special import hyp1f1, poch
from scipy.stats import beta as scipy_beta

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
    
    # 单机构风险度量
    def unconditional_var(self, idx, p_i, alpha):
        """利用精确逆函数计算单机构无条件 VaR"""
        prob_no_default = 1.0 - p_i
        
        # 保护机制：如果置信水平低于不违约概率，直接返回 0
        if alpha <= prob_no_default:
            return 0.0
            
        target_prob = (alpha - prob_no_default) / p_i
        epsilon_quantile = scipy_beta.ppf(target_prob, self.a[idx], self.b[idx])
        return self.u[idx] * epsilon_quantile

    # 蒙特卡洛模拟模块 Monte Carlo Simulation
    def simulate_portfolio_losses(self, p_cond_matrix, target_i=None, fixed_loss=None):
        """
        根据给定的条件违约概率矩阵模拟整个系统的条件损失
        p_cond_matrix 维度: (batch_size, N)
        """
        batch_size, N = p_cond_matrix.shape
        
        # 1. 模拟违约 D_j ~ Bernoulli(p_{j|Z})
        rand_unif = np.random.rand(batch_size, N)
        defaults = (rand_unif < p_cond_matrix).astype(float)
        
        # 2. 模拟违约损失率 \epsilon_j ~ Beta(a_j, b_j)
        # 借助 numpy 的 size 参数直接生成二维矩阵，彻底向量化
        lgd_sim = np.random.beta(self.a, self.b, size=(batch_size, N))
        
        # 3. 计算单体损失 L_j = u_j * D_j * \epsilon_j
        losses = self.u * defaults * lgd_sim
        
        # 4. 如果有目标机构处于确定危机状态，强制覆盖其损失
        if target_i is not None and fixed_loss is not None:
            losses[:, target_i] = fixed_loss
            
        # 返回每个场景下的系统总损失 (维度: batch_size,)
        return np.sum(losses, axis=1)


