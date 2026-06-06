#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:21:10 2026

@author: jaydenwang

功能：单因子依赖结构管理箱（已全面升级支持 Gumbel 因子耦合测度）
"""

import numpy as np
from scipy.stats import norm, t, chi2, skewnorm
from scipy.special import roots_genlaguerre
from scipy.special import gamma as gamma_func
from scipy.optimize import root_scalar

# 引入新创建的 Gumbel 联结函数
from gumbel_interplay import gumbel_copula_pdf
import config

np.random.seed(42)

class GaussianCopulaSingleFactor:
    """
    单因子高斯 Copula 依赖结构
    注：高斯架构下无冲击因子 Lambda，因此天然不受参数 xi 的影响，作为 Benchmark 保持原样。
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
        return norm.cdf(np.clip(num / den, -10, 10))
    
    
class TCopulaSingleFactor:
    """
    真正的单因子 t-Copula 依赖结构
    当 xi = 1 时退化为独立因子的传统 t-Copula，基于解析解 ppf 极速配置；
    当 xi > 1 时自动开启 2D 积分网格，将 Gumbel 依赖联合密度融入阈值 x_d 的数值反解。
    """
    def __init__(self, rho, p, nu):
        self.rho = np.array(rho)
        self.p = np.array(p)
        self.nu = nu
        self.x_d = np.zeros_like(self.p)
        
        # 读取配置中的 xi 参数
        self.xi = config.XI if hasattr(config, 'XI') else 1.0
        
        if self.xi == 1.0:
            # 独立状态下，资产边缘分布严格遵循标准 t 分布，直接解析解秒杀
            self.x_d = t.ppf(1 - self.p, df=self.nu)
        else:
            # 【核心升级点】相关状态下，资产边缘分布变形，必须使用 2D 求积网格进行数值反解
            N_GH, N_GL = 60, 60
            
            # 1. 构建系统因子 Z 网格 (标准正态分布测度)
            nodes_GH, weights_GH = np.polynomial.hermite.hermgauss(N_GH)
            Z_nodes = np.sqrt(2.0) * nodes_GH
            Z_weights = weights_GH / np.sqrt(np.pi)
            
            # 2. 构建冲击因子 Lambda 网格 (广义高斯-拉盖尔测度)
            nodes_GL, weights_GL = roots_genlaguerre(N_GL, alpha=self.nu / 2.0 - 1.0)
            Lambda_nodes = 2.0 * nodes_GL
            Lambda_weights = weights_GL / gamma_func(self.nu / 2.0)
            
            # 3. 前置计算全网格上的 Gumbel 相互依赖矩阵
            U_grid = norm.cdf(Z_nodes)[:, None]  # 形状 (N_GH, 1)
            V_grid = chi2.sf(Lambda_nodes, self.nu)[None, :]  # 形状 (1, N_GL)
            gumbel_matrix = gumbel_copula_pdf(U_grid, V_grid, self.xi)
            
            # 4. 逐个机构通过寻根法精准校准无条件违约概率
            for i in range(len(self.p)):
                rho_i = self.rho[i]
                p_i = self.p[i]
                
                def unconditional_prob(x):
                    Z_grid = Z_nodes[:, None]
                    L_grid = Lambda_nodes[None, :]
                    num = rho_i * Z_grid - x * np.sqrt(L_grid / self.nu)
                    den = np.sqrt(1 - rho_i**2)
                    p_cond = norm.cdf(np.clip(num / den, -10, 10))
                    # 乘入联合权重以及 gumbel 相互作用因子
                    return np.sum(p_cond * Z_weights[:, None] * Lambda_weights[None, :] * gumbel_matrix)
                
                res = root_scalar(lambda x: unconditional_prob(x) - p_i, bracket=[-10, 10], method='brentq')
                self.x_d[i] = res.root
        
    def conditional_default_prob(self, Z, Lambda):
        """
        计算给定 Z 和 Lambda 下的条件违约概率
        给定因子后条件独立，结构形式保持一致
        """
        Z = np.asarray(Z)
        Lambda = np.asarray(Lambda)
        sqrt_term = np.sqrt(Lambda / self.nu)
        
        if Z.ndim > 0 and Lambda.ndim > 0:
            Z_term = np.outer(Z, self.rho)
            Lambda_term = np.outer(sqrt_term, self.x_d)
            num = Z_term - Lambda_term
        else:
            num = self.rho * Z - self.x_d * sqrt_term
            
        den = np.sqrt(1 - self.rho**2)
        return norm.cdf(np.clip(num / den, -10, 10))


class SkewTCopulaSingleFactor:
    """
    单因子 Skew-t Copula 依赖结构 (升级版)
    系统性风险因子 Z ~ Skew-Normal(0, 1, gamma)
    共同冲击因子 Lambda ~ Chi^2(nu)
    两因子通过参数为 xi 的 Gumbel Copula 联结
    """
    def __init__(self, rho, p, nu, gamma_val):
        self.rho = np.array(rho)
        self.p = np.array(p)
        self.nu = nu
        self.gamma = gamma_val
        self.x_d = np.zeros_like(self.p)
        self.xi = config.XI if hasattr(config, 'XI') else 1.0
        
        N_GH, N_GL = 60, 60
        
        # 1. Z 的高斯-赫米特求积调整 (融入偏正态分布的 PDF：2 * \phi(z) * \Phi(\gamma z))
        nodes_GH, weights_GH = np.polynomial.hermite.hermgauss(N_GH)
        Z_nodes = np.sqrt(2.0) * nodes_GH
        Z_weights = (weights_GH / np.sqrt(np.pi)) * 2.0 * norm.cdf(self.gamma * Z_nodes)
        
        # 2. Lambda 的广义高斯-拉盖尔求积调整
        nodes_GL, weights_GL = roots_genlaguerre(N_GL, alpha=self.nu / 2.0 - 1.0)
        Lambda_nodes = 2.0 * nodes_GL
        Lambda_weights = weights_GL / gamma_func(self.nu / 2.0)
        
        # 3. 前置提取 Gumbel 相互作用测度矩阵
        # 注意：此处 Z 轴的累积分布函数必须采用一维偏正态分布 skewnorm.cdf
        U_grid = skewnorm.cdf(Z_nodes, self.gamma)[:, None]  # 形状 (N_GH, 1)
        V_grid = chi2.sf(Lambda_nodes, self.nu)[None, :]      # 形状 (1, N_GL)
        gumbel_matrix = gumbel_copula_pdf(U_grid, V_grid, self.xi)
        
        # 4. 逐个机构通过寻根法反解违约门槛 x_i^d
        for i in range(len(self.p)):
            rho_i = self.rho[i]
            p_i = self.p[i]
            
            def unconditional_prob(x):
                # 建立二维因子网格
                Z_grid = Z_nodes[:, None]
                L_grid = Lambda_nodes[None, :]
                num = rho_i * Z_grid - x * np.sqrt(L_grid / self.nu)
                den = np.sqrt(1 - rho_i**2)
                p_cond = norm.cdf(np.clip(num / den, -10, 10))
                # 乘入联合网格及 Gumbel 密度校正矩阵
                return np.sum(p_cond * Z_weights[:, None] * Lambda_weights[None, :] * gumbel_matrix)
            
            res = root_scalar(lambda x: unconditional_prob(x) - p_i, bracket=[-10, 10], method='brentq')
            self.x_d[i] = res.root

    def conditional_default_prob(self, Z, Lambda):
        """
        计算给定 Z 和 Lambda 条件下的条件违约概率
        """
        Z = np.asarray(Z)
        Lambda = np.asarray(Lambda)
        num = self.rho * Z - self.x_d * np.sqrt(Lambda / self.nu)
        den = np.sqrt(1 - self.rho**2)
        return norm.cdf(np.clip(num / den, -10, 10))