#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 21:28:08 2026

@author: jaydenwang
"""


import numpy as np

def gumbel_copula_pdf(u, v, xi):
    """
    计算参数为 xi 的 Gumbel Copula 在 (u, v) 处的联合概率密度 (PDF)
    支持向量化/矩阵广播输入
    """
    if xi == 1.0:
        return np.ones_like(u) if isinstance(u, np.ndarray) else 1.0
        
    # 数值截断保护，防止 log(0) 或 log(1) 产生 NaN
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    v = np.clip(v, 1e-12, 1.0 - 1e-12)
    
    x = -np.log(u)
    y = -np.log(v)
    
    A = x**xi + y**xi
    # Gumbel CDF 表达式 (保留作参考或检查)
    # C = np.exp(-A**(1.0 / xi))
    
    # 严格推导的标准 Gumbel PDF 展开式
    term_C = np.exp(-A**(1.0 / xi))
    term_uv = u * v
    term_A1 = A**(1.0 / xi + -2.0)
    term_xy = (x * y)**(xi - 1.0)
    term_bracket = A**(1.0 / xi) + xi - 1.0
    
    pdf = (term_C / term_uv) * term_A1 * term_xy * term_bracket
    return pdf