#!/usr/bin/env python3
"""统一计算入口：选择方法和指标后直接运行；其余参数在 config.py 修改。"""

import config
from calculation_runner import run_calculation


def main(cfg=config):
    # 只需修改下面 return 行中的两个字符串（均为小写）。
    #
    # 第一个字符串：计算方法
    #   "spa"     : Saddlepoint Approximation
    #   "pmc"     : Plain Monte Carlo
    #   "ce_rqmc" : Cross-Entropy + Randomized Quasi-Monte Carlo
    #
    # 第二个字符串：风险指标
    #   "covar"  : 只计算 CoVaR
    #   "coes"   : 计算并输出 CoVaR + CoES
    #   "ecovar" : 只计算 E-CoVaR
    #   "ecoes"  : 计算并输出 E-CoVaR + E-CoES
    #
    # 示例：
    #   run_calculation("spa", "coes", cfg)      -> SPA 的 CoVaR + CoES
    #   run_calculation("pmc", "ecoes", cfg)     -> PMC 的 E-CoVaR + E-CoES
    #   run_calculation("ce_rqmc", "ecovar", cfg)-> CE-RQMC 的 E-CoVaR
    #
    # 等式/不等式条件：在 config.py 修改 CONDITION_MODE：
    #   "equality" / "inequality"
    # 模型、alpha/beta、积分精度、带宽、模拟预算等也都在 config.py。
    # 三种方法都输出 CPU / wall 时间；PMC、CE-RQMC 另输出各指标的 SE。
    return run_calculation("spa", "ecoes", cfg)


if __name__ == "__main__":
    main()
