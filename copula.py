#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:21:10 2026

@author: jaydenwang

功能：单因子依赖结构管理箱（已全面升级支持 Gumbel 因子耦合测度）
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm, t, chi2, skewnorm
from scipy.optimize import root_scalar

# The same Rosenblatt transform is used for threshold calibration and for the
# production factor integral.  This avoids a model-changing mismatch between a
# density-weighted Hermite--Laguerre grid and a probability-space grid.
from gumbel_interplay import gumbel_conditional_inverse


def _uniform_rule(order):
    nodes, weights = leggauss(int(order))
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _calibrate_dependent_thresholds(
    rho,
    marginal_pd,
    nu,
    xi,
    z_ppf,
    n_z,
    n_lambda,
):
    """Calibrate default thresholds on the production Rosenblatt measure."""
    if int(n_z) < 8 or int(n_lambda) < 8:
        raise ValueError("threshold-calibration orders must be at least eight")
    u_nodes, u_weights = _uniform_rule(n_z)
    conditional_nodes, conditional_weights = _uniform_rule(n_lambda)
    uu, qq = np.meshgrid(u_nodes, conditional_nodes, indexing="ij")
    survival = gumbel_conditional_inverse(uu, qq, xi)
    lambda_grid = chi2.isf(survival, nu)
    z_grid = np.broadcast_to(z_ppf(u_nodes)[:, None], lambda_grid.shape)
    weights = (u_weights[:, None] * conditional_weights[None, :]).ravel()
    z_flat = z_grid.ravel()
    sqrt_lambda = np.sqrt(lambda_grid.ravel() / nu)

    thresholds = np.empty_like(np.asarray(marginal_pd, dtype=float))
    for i, (rho_i, p_i) in enumerate(zip(rho, marginal_pd)):
        denominator = np.sqrt(1.0 - rho_i**2)

        def unconditional_prob(threshold):
            argument = (
                rho_i * z_flat - threshold * sqrt_lambda
            ) / denominator
            return np.dot(weights, norm.cdf(np.clip(argument, -10.0, 10.0)))

        result = root_scalar(
            lambda threshold: unconditional_prob(threshold) - p_i,
            bracket=(-10.0, 10.0),
            method="brentq",
            xtol=1e-12,
        )
        thresholds[i] = result.root
    return thresholds

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
        
    # def conditional_default_prob(self, Z):
    #     """
    #     计算在给定单因子系统风险 Z 下的条件违约概率 p_{i|\Theta}
    #     """
    #     num = self.rho * Z - self.x_d
    #     den = np.sqrt(1 - self.rho**2)
    #     return norm.cdf(np.clip(num / den, -10, 10))
    
    def conditional_default_prob(self, Z):
        """
        计算在给定单因子系统风险 Z 下的条件违约概率 p_{i|\Theta}
        """
        Z = np.asarray(Z)
        
        # 兼容向量化输入 (例如 MC 引擎一次传入 2000 条路径)
        if Z.ndim > 0:
            num = np.outer(Z, self.rho) - self.x_d
        else:
            # 兼容标量输入 (例如数值积分引擎一次传入 1 个节点)
            num = self.rho * Z - self.x_d
            
        den = np.sqrt(1 - self.rho**2)
        return norm.cdf(np.clip(num / den, -10, 10))
    
    
class TCopulaSingleFactor:
    """
    真正的单因子 t-Copula 依赖结构
    当 xi = 1 时退化为独立因子的传统 t-Copula，基于解析解 ppf 极速配置；
    当 xi > 1 时自动开启 2D 积分网格，将 Gumbel 依赖联合密度融入阈值 x_d 的数值反解。
    """
    def __init__(
        self,
        rho,
        p,
        nu,
        xi=1.0,
        calibration_n_z=64,
        calibration_n_lambda=64,
    ):
        self.rho = np.array(rho)
        self.p = np.array(p)
        self.nu = nu
        self.x_d = np.zeros_like(self.p)
        
        self.xi = float(xi)
        if self.xi < 1.0:
            raise ValueError("xi must be at least one")
        
        if self.xi == 1.0:
            # 独立状态下，资产边缘分布严格遵循标准 t 分布，直接解析解秒杀
            self.x_d = t.ppf(1 - self.p, df=self.nu)
        else:
            self.x_d = _calibrate_dependent_thresholds(
                self.rho,
                self.p,
                self.nu,
                self.xi,
                norm.ppf,
                calibration_n_z,
                calibration_n_lambda,
            )
        
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
    def __init__(
        self,
        rho,
        p,
        nu,
        gamma_val,
        xi=1.0,
        calibration_n_z=64,
        calibration_n_lambda=64,
    ):
        self.rho = np.array(rho)
        self.p = np.array(p)
        self.nu = nu
        self.gamma = gamma_val
        self.x_d = np.zeros_like(self.p)
        self.xi = float(xi)
        if self.xi < 1.0:
            raise ValueError("xi must be at least one")
        
        self.x_d = _calibrate_dependent_thresholds(
            self.rho,
            self.p,
            self.nu,
            self.xi,
            lambda probability: skewnorm.ppf(probability, self.gamma),
            calibration_n_z,
            calibration_n_lambda,
        )

    # def conditional_default_prob(self, Z, Lambda):
    #     """
    #     计算给定 Z 和 Lambda 条件下的条件违约概率
    #     """
    #     Z = np.asarray(Z)
    #     Lambda = np.asarray(Lambda)
    #     num = self.rho * Z - self.x_d * np.sqrt(Lambda / self.nu)
    #     den = np.sqrt(1 - self.rho**2)
    #     return norm.cdf(np.clip(num / den, -10, 10))
    
    
    def conditional_default_prob(self, Z, Lambda):
        """
        计算给定 Z 和 Lambda 条件下的条件违约概率
        """
        Z = np.asarray(Z)
        Lambda = np.asarray(Lambda)
        sqrt_term = np.sqrt(Lambda / self.nu)
        
        # 兼容向量化输入
        if Z.ndim > 0 and Lambda.ndim > 0:
            Z_term = np.outer(Z, self.rho)
            Lambda_term = np.outer(sqrt_term, self.x_d)
            num = Z_term - Lambda_term
        else:
            num = self.rho * Z - self.x_d * sqrt_term
            
        den = np.sqrt(1 - self.rho**2)
        return norm.cdf(np.clip(num / den, -10, 10))
    
    
    
    
