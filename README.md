# What Drives the Price of a Used Car?

An analysis of 426,880 used-car listings using CRISP-DM method, with inventory recommendations for a used car dealership.

## Project summary

A dealership wants to know what consumers value in a used car. I framed this as a supervised regression problem: predict `log(price)` from vehicle attributes, using interpretable linear models (Linear Regression, Ridge, Lasso) so that every coefficient can be translated into a percentage effect on price.

| Result | Value |
|---|---|
| Cleaned dataset | 313,596 of 426,880 listings (73%) |
| Cross-validated R² (5-fold, training set) | 0.779 |
| Test R² (held-out 20%) | 0.785 |
| Test MAE | about \$4,700 (median error about 20%) |
| Final model | Ridge, alpha = 1 (tuned by grid search; Lasso tied) |

## Key findings

- **Age:** each extra year lowers price by about 7%.
- **Mileage:** each extra 10,000 miles lowers price by about 4%, on top of the age effect. A car 5 years older with 50,000 more miles is expected to be about 43% cheaper.
- **Condition and title:** *fair* and salvage cars sell about 40% lower than good; excellent about 10% higher. Salvage titles cost about 29% and rebuilt titles about 23%.
- **Body type :** trucks +36%, pickups +34%, SUVs +13%.
- **Drivetrain and engine:** 4WD +30% and RWD +24% (vs. FWD); 8-cylinder +23%, 4-cylinder about 18% lower (vs. 6-cylinder).
- **Brand:** Lexus +38%, Audi +26%, Mercedes-Benz +21%, Toyota +17%; Mitsubishi −25%, Kia −20%, Chrysler −20%.
- **Fuel:** diesel about +100% (largely heavy-duty trucks); hybrid about level; electric about 12% lower once age, mileage and body type are accounted for.
- **Transmission:** raw prices for manuals are lower (they are older), but manual is about 15% higher than a comparable automatic. There is no evidence of an automatic premium.

## Recommendations for the dealership

1. Buy on age and mileage first; prefer lower-mileage examples within a model year.
2. Avoid salvage, rebuilt and "fair" cars unless the purchase discount covers the 20-40% markdown.
3. Lean toward trucks, pickups, SUVs and 4WD vehicles; compact 4-cylinder sedans carry the lowest prices.
4. Favor Toyota, Lexus, Audi and Mercedes-Benz for premium resale value.
5. Do not pay extra for hybrid or electric vehicles based on this data.
6. Do not assume automatics carry a premium.
7. Document condition on every listing (about 41% of listings omit it).

## Limitations and next steps

- Prices are asking prices, and there is no sales-speed or margin data, so higher price does not always mean higher profit.
- The model finds associations, not causes (for example, diesel is mixed with body type).
- Next steps: add sold-price and days-to-sell data, add model-level and regional information, try non-linear models and refit regularly on new listings.

## Methodology (CRISP-DM)

1. **Business understanding:** reframed as regression on `log(price)` with interpretable coefficients.
2. **Data understanding:** missing-value chart, price distribution (raw vs log), and quantified quality issues: 7.7% of prices are \$0, 13.2% of rows are duplicates, and 72% of `size` is missing.
3. **Data preparation:** removed duplicates, kept prices \$1k-\$100k, model years 1995-2021 and odometer 100-400k miles (row count logged at each step), kept missing categories as `unknown` instead of mode-filling, dropped identifiers and high-cardinality columns (`id`, `VIN`, `model`, `region`), engineered `car_age`, and used a scikit-learn `Pipeline` (scaling and one-hot encoding with explicit reference categories) to avoid data leakage.
4. **Modeling:** baseline plus Linear Regression, Ridge and Lasso, with 5-fold cross-validation and grid search over `alpha` for Ridge and Lasso.
5. **Evaluation:** R² (model selection), MAE in dollars (business reporting) and RMSE on log-price, with the rationale for each documented. Residual diagnostics and error by price band are included. Coefficients are converted to percentage effects.
6. **Deployment:** findings and recommendations written for a non-technical audience (Section 7 of the notebook).

## Repository structure

```
.
├── README.md
├── used_car_price_analysis.ipynb
└── data/
    └── vehicles.csv
```

## How to run

Requires Python 3.9+ with `pandas`, `numpy`, `matplotlib`, `seaborn` and `scikit-learn`. Put `vehicles.csv` in a `data/` folder next to the notebook, then run all cells.
