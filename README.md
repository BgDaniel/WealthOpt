# WealthOpt — stochastic retirement control and consumption optimization

`WealthOpt` simulates a risky asset (GBM), a savings account, monthly cashflows (salary, rent, pension),
and finds daily optimal controls (withdrawal and transfers) that maximize expected lifetime utility of consumption using dynamic programming.

## Features
- GBM path generator for risky asset
- Cashflow engine (salary, rent, pension)
- Daily control actions: withdraw consumption, transfer between accounts
- Bellman equation solver (discrete dynamic programming)
- Monte Carlo policy evaluation and plotting utilities

## Quick start
TBD: installation instructions and code examples.

## Roadmap
- v0.x: basic DP solver and path generator
- v1.0: interpolation & improved solvers
- v2.0: advanced A.D.P. and extensions (transaction costs, taxes)

## License
MIT