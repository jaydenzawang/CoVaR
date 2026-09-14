#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:21:23 2026

@author: jaydenwang

功能：投资组合损失模型，包含 LGD 的 Beta 分布 MGF 计算
采用高密度三次样条插值；均值附近的标量鞍点使用解析 MGF，避免 LR 公式放大插值误差。
"""

import numpy as np
from scipy.special import hyp1f1, poch
from scipy.interpolate import CubicSpline

class PortfolioLoss:
    """
    投资组合损失模型，采用样条与均值附近解析求值的 Beta MGF。
    """
    def __init__(self, u, a, b, s_min=-25.0, s_max=25.0, grid_size=1501, exact_mgf_radius=0.15):
        self.u = np.asarray(u, dtype=float)
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        if not (self.u.ndim == self.a.ndim == self.b.ndim == 1):
            raise ValueError("u, a and b must be one-dimensional arrays")
        if not (len(self.u) == len(self.a) == len(self.b)):
            raise ValueError("u, a and b must have the same length")
        if np.any(self.u <= 0) or np.any(self.a <= 0) or np.any(self.b <= 0):
            raise ValueError("exposures and beta shape parameters must be positive")
        if not (s_min < 0 < s_max) or grid_size < 100:
            raise ValueError("the interpolation grid must contain zero and be sufficiently dense")
        
        # =====================================================================
        # 【策略一优化】前置构建 s 因子空间高密度三次样条插值矩阵
        # =====================================================================
        # 隐患修复(6): 网格区间与 spa.py 鞍点截断 [-20, 20] 对齐, 两侧各留 5 的样条缓冲,
        # 避免网格端点条件污染 [19,20] 段系数; 加密到 1500 点提升有效区间分辨率。
        self.s_grid = np.linspace(float(s_min), float(s_max), int(grid_size))
        # Make the origin an exact interpolation knot.  K'(0) is the conditional
        # mean used by both the saddle equation and the stop-loss mean limit.
        self.s_grid[np.argmin(np.abs(self.s_grid))] = 0.0
        
        # 针对 spa.py 所需的 CGF 0 阶至 4 阶导数，前置一次性批量计算全组合机构的合流超几何函数
        # 结果矩阵形状均为 (1000, N)，完全保留了异质组合机构的资产敞口及参数特性
        hyp_k0 = np.array([hyp1f1(self.a, self.a + self.b, s * self.u) for s in self.s_grid])
        self.spline_k0 = CubicSpline(self.s_grid, hyp_k0, axis=0, extrapolate=False)
        
        hyp_k1 = np.array([hyp1f1(self.a + 1, self.a + self.b + 1, s * self.u) for s in self.s_grid])
        self.spline_k1 = CubicSpline(self.s_grid, hyp_k1, axis=0, extrapolate=False)
        
        hyp_k2 = np.array([hyp1f1(self.a + 2, self.a + self.b + 2, s * self.u) for s in self.s_grid])
        self.spline_k2 = CubicSpline(self.s_grid, hyp_k2, axis=0, extrapolate=False)
        
        hyp_k3 = np.array([hyp1f1(self.a + 3, self.a + self.b + 3, s * self.u) for s in self.s_grid])
        self.spline_k3 = CubicSpline(self.s_grid, hyp_k3, axis=0, extrapolate=False)
        
        hyp_k4 = np.array([hyp1f1(self.a + 4, self.a + self.b + 4, s * self.u) for s in self.s_grid])
        self.spline_k4 = CubicSpline(self.s_grid, hyp_k4, axis=0, extrapolate=False)
        
        self.exact_mgf_radius = float(exact_mgf_radius)
        if not np.isfinite(self.exact_mgf_radius) or self.exact_mgf_radius <= 0:
            raise ValueError("exact_mgf_radius must be finite and positive")
        self._max_exposure = float(np.max(self.u))
        self.expected_epsilon = self.a / (self.a + self.b)
        # Parameters are fixed for this model context. Keep multiplication order
        # unchanged: coefficient * spline value * exposure power.
        self._derivative_coef = {k: poch(self.a, k) / poch(self.a + self.b, k)
                                 for k in (1, 2, 3, 4)}
        self._exposure_power = {k: self.u ** k for k in (1, 2, 3, 4)}
        
    def mgf_epsilon(self, s):
        """计算 MGF；均值附近的标量鞍点直接调用 hyp1f1。"""
        self._validate_s(s)
        # The second-order LR formula contains cancelling inverse powers of w.
        # Cubic-spline error in K(s), although tiny absolutely, is amplified near
        # the mean. Use the analytic MGF and its consistent derivatives there.
        if np.ndim(s) == 0 and abs(s) * self._max_exposure < self.exact_mgf_radius:
            return hyp1f1(self.a, self.a + self.b, s * self.u)
        return self.spline_k0(s)
        
    def mgf_epsilon_deriv(self, s, k=1):
        """计算与 MGF 求值方式一致的各阶导数，包含敞口链式法则。"""
        self._validate_s(s)
        if k not in (1, 2, 3, 4):
            raise ValueError("only derivatives of order 1 through 4 are supported")
        coef = self._derivative_coef[k]
        if np.ndim(s) == 0 and abs(s) * self._max_exposure < self.exact_mgf_radius:
            return coef * hyp1f1(self.a + k, self.a + self.b + k, s * self.u) * self._exposure_power[k]
        if k == 1:
            return coef * self.spline_k1(s) * self.u
        elif k == 2:
            return coef * self.spline_k2(s) * self._exposure_power[2]
        elif k == 3:
            return coef * self.spline_k3(s) * self._exposure_power[3]
        elif k == 4:
            return coef * self.spline_k4(s) * self._exposure_power[4]
        raise AssertionError("unreachable derivative order")

    def _validate_s(self, s):
        values = np.asarray(s)
        if np.any(values < self.s_grid[0]) or np.any(values > self.s_grid[-1]):
            raise ValueError(
                f"s={s} lies outside the validated interpolation interval "
                f"[{self.s_grid[0]}, {self.s_grid[-1]}]"
            )

    def A_j(self, s, p_cond):
        """计算 A_j(s|\\theta) (返回维度为 N 的数组)"""
        return 1 - p_cond + p_cond * self.mgf_epsilon(s)
        
    def A_j_deriv(self, s, p_cond, k):
        """计算 A_j 的 k 阶导数 (返回维度为 N 的数组)
        注意: mgf_epsilon_deriv(s,k) 已含链式因子 u^k, 即已等于 u_j^k * M_eps^(k)(s*u_j),
        恰为论文中的 A_j^(k)/p_{j|Theta}。此处只需乘条件违约概率 p_cond,
        切勿再乘 u^k, 否则 u_i != 1 时会静默给出错误的 CGF 导数。
        """
        return p_cond * self.mgf_epsilon_deriv(s, k)
