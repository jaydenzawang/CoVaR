#!/usr/bin/env python3
"""
Main entry point: Select the method and measure, then run; set other parameters in config.py.
"""

import config
from calculation_runner import run_calculation


def main(cfg=config):
    # Edit only the two strings in the return line below (both lowercase).
    #
    # First string: Method
    #   "spa"     : Saddlepoint Approximation
    #   "pmc"     : Plain Monte Carlo
    #   "ce_rqmc" : Cross-Entropy + Randomized Quasi-Monte Carlo
    #
    # Second string: Risk measure
    #   "coes"   : Compute and print CoVaR + CoES
    #   "ecoes"  : Compute and print E-CoVaR + E-CoES
    #
    # Examples:
    #   run_calculation("spa", "coes", cfg)      -> CoVaR + CoES via SPA
    #   run_calculation("pmc", "ecoes", cfg)     -> E-CoVaR + E-CoES via PMC
    #   run_calculation("ce_rqmc", "ecovar", cfg)-> E-CoVaR via CE-RQMC
    #
    # Equality/inequality condition: Set CONDITION_MODE in config.py:
    #   "equality" / "inequality"
    # Set the model, alpha/beta, quadrature accuracy, bandwidth, and sample sizes in config.py too.
    # All three methods print CPU / wall time; PMC and CE-RQMC also print SE for each measure.
    
    
    
    return run_calculation("spa", "coes", cfg)


if __name__ == "__main__":
    main()
