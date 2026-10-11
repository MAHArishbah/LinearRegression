# Advertising Spend → Sales: A Diagnostic-Driven Linear Regression

Predicting `sales` from TV, radio, and newspaper advertising spend on the classic
200-row Advertising dataset.

The dataset is easy. The point of this project is not the final R² — it is that
**every modelling decision is forced by a diagnostic from the previous step**, and
that the model which wins on cross-validation is rejected in favour of one that is
equally accurate but actually identifiable and extrapolatable.

**Final model** (held-out test set, 40 rows):

| | |
|---|---|
| Test R² | **0.996** |
| Test RMSE | **0.360** |
| Test MSE | 0.130 |
| Target std | 5.109 |

```
sales ≈ -0.544 + 2.122·log(tv + 1) + 0.0485·radio + 0.00100·(tv × radio)
```

Three features. No regularisation. No polynomial basis.

---

## The path to it

Protocol: 80/20 split (`random_state=42`), 5-fold shuffled `KFold` for all model
selection, test set touched exactly once at the end.

| Step | Model | CV MSE | CV R² (per fold) |
|---|---|---|---|
| 0 | `DummyRegressor(mean)` | 26.68 | — |
| 1 | OLS on raw features | 2.88 | 0.870 ± 0.072 |
| 2 | Lasso + polynomial, 1-SE rule (deg 2, α=0.028) | 0.55 | 0.978 ± 0.022 |
| 3 | **Targeted features** (above) | **0.10** | **0.996 ± 0.001** |

29× better than the linear baseline, 271× better than predicting the mean.

### Why each step happened

**EDA → expect non-linearity.** TV's relationship with sales is concave and flattens
near spend 300 (diminishing returns). Radio looks linear. Newspaper has outliers in
the predictor, but the corresponding sales values are unremarkable — marginal, not
influential. VIF on raw features is everywhere < 5, so the raw design matrix is fine;
whatever collinearity appears later is something *I* will have introduced.

**Baseline residuals → name the missing terms.** The residual-vs-fitted plot curves.
Residuals against TV form a downward U, and all sit above zero — a concave transform
of TV is needed. Residuals against radio form an upward U despite radio's scatter
being straight, which is the signature of a missing *interaction*, not a missing
power. Residuals against newspaper are a structureless cloud. The learning curve
shows train and validation error both high and converged: high bias, so the fix is
features, not more data or more regularisation.

**Polynomial Lasso → accurate, but for the wrong reasons.** Grid search over degree
1–4 and 30 alphas picks degree 3, α=0.0016, CV MSE 0.414 — the best number in the
whole notebook. Its TV coefficients are +6.89, −9.19, +4.12.

That triple is the finding this project is really about. A modest saturating curve is
being expressed through near-total cancellation between three huge terms, because
`tv`, `tv²` and `tv³` evaluated over 0–300 are nearly parallel vectors. A cubic lives
in the 4-dimensional span of {1, tv, tv², tv³}; monomials are a *bad set of axes* for
that space, and expressing a point in a near-parallel basis requires extreme
coordinates that cancel. The design matrix's condition number is 109. Three
consequences:

- **No interpretability.** Bootstrap-refit and the predictions barely move while the
  coefficients swing wildly. The combination is identified; the individual
  coefficients are not.
- **Extrapolation failure.** With −9.19 on `tv²` and +4.12 on `tv³`, the second
  derivative is `2(−9.19) + 6(4.12)·tv`, negative at low TV and turning positive at
  tv ≈ 0.74 in scaled units. The fitted curve saturates in-sample and then
  *accelerates upward* past the data range — exactly backwards for ad spend.
- **Unstable sparsity.** Lasso's choice of which collinear term to zero is arbitrary
  under collinearity.

Cross-validation cannot see any of this: it measures interpolation, and interpolation
is the one thing the cubic does well.

**1-SE rule → shrink the damage.** Taking the largest α within one standard error of
the CV minimum drops to degree 2, α=0.028, CV MSE 0.547, condition number 13.5. The
surviving coefficients (`tv radio` 3.56, `tv` 3.48, `tv²` −1.97, `radio` 0.39) now at
least describe the shape the EDA predicted. All three newspaper terms are zeroed.
Better, but still a basis chosen for convenience rather than for the mechanism.

**Targeted features → the same shape, stated honestly.** `log(tv+1)` encodes
saturation directly, in a basis where each coefficient is independently meaningful,
and `tv × radio` encodes the interaction the baseline residuals asked for. Newspaper
is dropped. The result beats the polynomial on CV by 5× with a third of the
parameters, and every coefficient is readable:

- `log(tv+1)` = 2.122 → **+0.021 sales per 1% more TV spend**, or +1.47 per doubling.
- `radio` = 0.0485 → +0.049 sales per unit of radio spend at zero TV.
- `tv × radio` = 0.0010 → radio's return rises with TV spend. The channels amplify
  each other.

All three are significant at p < 0.001 under HC3-robust standard errors
(R² = 0.997 in-sample, F = 1.3e4).

---

## Diagnostics on the final model

**Influence.** Row 130 is the one genuinely troublesome observation: Cook's D = 1.28
in the polynomial model with studentised residual −11.1 but hat value only 0.085 — it
moves the fit through a large *residual*, not through leverage. The targeted model
absorbs most of this (Cook's D falls to 0.52). Row 155 follows a similar pattern at
lower magnitude; after those two, Cook's D drops by an order of magnitude. Refitting
without them shifts coefficients 3–17%.

Note that row 101 shows up in univariate IQR screening but never in influence
diagnostics — feature screening and influence answer different questions, and neither
substitutes for the other.

**Heteroscedasticity — unresolved, and stated as such.** Breusch–Pagan rejects
constant variance (p = 0.0042 for the targeted model, p = 0.0010 for the polynomial).
Dropping rows 130 and 155 makes it *worse* (p = 3.9e-6), which rules out the obvious
explanation: the non-constant variance is a property of the data, not an artefact of
two bad rows. Coefficient estimates remain unbiased and the HC3 standard errors above
are valid. **The prediction intervals in the notebook are not** — they rest on a
pooled σ and should be read as approximate.

---

## Files

```
Advertising.csv                                  200 rows: tv, radio, newspaper, sales
AdvertisemaentSpendPrediction_Version3.ipynb     current analysis — read this one
AdvertisemaentSpendPrediction_Version1.ipynb     earlier iterations, kept for history
AdvertisemaentSpendPrediction_Version2.ipynb
ADV_SPEND_Prediction.joblib                      final model, refit on all 200 rows
```

The saved artefact is a dict holding the fitted pipeline, the expected column order,
and the sklearn version it was trained under.

Requires numpy, pandas, matplotlib, seaborn, scipy, scikit-learn, statsmodels, joblib.

---

## Known issue: the pickle is not portable

`targeted_features` is defined at notebook scope and referenced by the pipeline's
`FunctionTransformer`, so `joblib.load` fails in a fresh process unless that exact
function is already importable under the same qualified name. Fixing it properly means
one of: moving the function into an importable module, replacing it with a custom
`BaseEstimator`/`TransformerMixin` class, or serialising with cloudpickle.

## If this were taken further

- Weighted least squares or a variance-stabilising transform on the target, to address
  the Breusch–Pagan result rather than just reporting robust errors.
- Orthogonal polynomials as a cleaner demonstration of the basis argument in Step 2 —
  same function space, perpendicular axes, coefficients that are independent
  projections.
- Bootstrap coefficient intervals, which would make the polynomial model's
  non-identifiability visible rather than argued.
- The real limit is `n = 200` with a single split. The ± on the per-fold CV scores is
  the honest uncertainty here, and a repeated / nested CV would report it better.
