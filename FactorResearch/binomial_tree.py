Implements a simplified Black-Derman-Toy (BDT-style) lognormal
recombining short-rate tree calibrated to a spot curve.

Features
--------
1. Bootstrap spot rates from par rates
2. Build arbitrage-free short-rate tree
3. Price option-free bonds
4. Price callable / putable bonds
5. Compute Option-Adjusted Spread (OAS)
6. Compute embedded option value

Author Notes
-------------
- Uses continuously-compounded rates internally
- Uses equal risk-neutral probabilities (0.5 / 0.5)
