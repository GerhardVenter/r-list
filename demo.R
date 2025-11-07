1. Simple Linear Regression (SLR)
   Model: E(y) = β0 + β1x
   R code:
   fit <- lm(y ~ x)
   summary(fit)
   anova(fit)
   Interpretation:

* coef(fit): β0, β1
* summary(fit)$r.squared: R2
* summary(fit)$sigma: s = √MSE
* anova(fit): F-test
* confint(fit): confidence intervals
* predict(fit, newdata, interval="confidence" or "prediction")

2. Multiple Linear Regression (MLR)
   Model: E(y) = β0 + β1x1 + β2x2 + ... + βkxk
   R code:
   fit <- lm(y ~ x1 + x2 + x3)
   summary(fit)
   anova(fit)
   Partial F-test:
   reduced <- lm(y ~ x1 + x2)
   full <- lm(y ~ x1 + x2 + x3)
   anova(reduced, full)
   If p < 0.05 → added variable is significant.

3. Models with Indicator (Dummy) Variables
   Used for categorical predictors.
   Example:
   Model: E(y) = β0 + β1x + β2D
   fit <- lm(y ~ x + D)  # D = 0/1 indicator
   If multiple categories:
   fit <- lm(y ~ factor(Group))  # R auto-generates dummies

4. Interaction Models
   Model: E(y) = β0 + β1x + β2D + β3xD
   R expands x * D to x + D + x:D
   fit <- lm(y ~ x * D)
   Interpretation:
   β0 = intercept for D=0
   β1 = slope for D=0
   β2 = vertical shift (difference at x=0)
   β3 = slope difference between groups

5. Polynomial Regression
   Model: E(y) = β0 + β1x + β2x^2
   fit <- lm(y ~ x + I(x^2))
   I() prevents R from interpreting ^ as formula operator.

6. Piecewise (Segmented) Regression
   (a) Continuous two-piece (slope change only)
   E(y) = β0 + β1x + β2(x - k)I(x>k)
   x2 <- ifelse(x > k, x - k, 0)
   fit <- lm(y ~ x + x2)

(b) Discontinuous two-piece (slope + level change)
E(y) = β0 + β1x + β2(x - k)I(x>k) + β3I(x>k)
x2 <- ifelse(x > k, x - k, 0)
I2 <- as.numeric(x > k)
fit <- lm(y ~ x + x2 + I2)

(c) Three-piece continuous
E(y) = β0 + β1x + β2(x - k1)I(x>k1) + β3(x - k2)I(x>k2)
x2 <- ifelse(x > k1, x - k1, 0)
x3 <- ifelse(x > k2, x - k2, 0)
fit <- lm(y ~ x + x2 + x3)

7. Useful Helper Functions
   coef(fit)              # coefficients
   fitted(fit)            # fitted (predicted) y-values
   resid(fit)             # residuals
   df.residual(fit)       # degrees of freedom
   sum(resid(fit)^2)      # SSE
   deviance(fit)/df.residual(fit)  # MSE
   summary(fit)$sigma     # standard error of estimate
   summary(fit)$r.squared # R2
   summary(fit)$adj.r.squared # Adjusted R2
   anova(fit)             # ANOVA table
   anova(model1, model2)  # partial F-test
   predict(fit, newdata, interval="confidence" or "prediction")
   plot(fit)              # residual diagnostics

8. Statistical Testing Summary
   Test / R equivalent / Null hypothesis
   Single t-test: summary(fit) / H0: βi = 0
   Overall F-test: summary(fit) or anova(fit) / H0: all slopes = 0
   Partial F-test: anova(reduced, full) / H0: added terms = 0
   Piecewise slope change: t-test on β2 / H0: β2 = 0
   Discontinuity: t-test on β3 / H0: β3 = 0
   Joint slope + discontinuity: partial F with 2 df / H0: β2 = β3 = 0

9. General R Commands and Workspace Management
   Clear console:
   Ctrl + L  (keyboard shortcut)

General Commands:

* Clear environment variables: rm(list = ls())

* Clear plots in RStudio:
  if (dev.cur() > 1) dev.off()  # close current plot window
  while (!is.null(dev.list())) dev.off()  # close all plots


STAS202 FORMULA SHEET

y = β0 + β1X1 + β2X2 + ... + βkXk + ε

b1 = SSxy / SSxx

SSxy = Σ(xi - x̄)(yi - ȳ) = Σ(xi * yi) - n * x̄ * ȳ

SSxx = Σ(xi - x̄)² = Σ(xi²) - n(x̄)²

SSyy = Σ(yi - ȳ)² = Σ(yi²) - n(ȳ)²

SSyy = SSE / (1 - R²)

b0 = ȳ - b1x̄

s² = SSE / (n - 2)

SSE = Σ(yi - ŷi)² = SSyy - b1SSxy

σ(b1) = σ / √SSxx

t = b1 / s(b1) = b1 / (s / √SSxx)

Confidence interval for b1:
b1 ± t(α/2) * s(b1)

r = SSxy / √(SSxx * SSyy)

t = r√(n - 2) / √(1 - r²)

r² = (SSyy - SSE) / SSyy = 1 - SSE / SSyy

σ(ŷ) = σ * √(1/n + (xp - x̄)² / SSxx)

σ(y - ŷ) = σ * √(1 + 1/n + (xp - x̄)² / SSxx)

Confidence interval for mean response:
ŷ ± t(α/2) * s * √(1/n + (xp - x̄)² / SSxx)

Prediction interval for individual response:
ŷ ± t(α/2) * s * √(1 + 1/n + (xp - x̄)² / SSxx)

b1 = (Σxi * yi) / (Σxi²)

s² = SSE / (n - 1)

SSE = Σyi² - b1Σxi * yi

s(b1) = s / √(Σxi²)

s(ŷ) = s * (xp / √(Σxi²))

s(y - ŷ) = s * √(1 + xp² / Σxi²)

t = (b1 - 0) / s(b1) = b1 / (s / √(Σxi²))

Confidence interval for slope:
b1 ± t(α/2) * s(b1) = b1 ± t(α/2) * (s / √(Σxi²))

Confidence interval for mean response:
ŷ ± t(α/2) * s(ŷ) = ŷ ± t(α/2) * s * (xp / √(Σxi²))

Prediction interval for individual response:
ŷ ± t(α/2) * s(y - ŷ) = ŷ ± t(α/2) * s * √(1 + xp² / Σxi²)

t = bi / s(bi)

F = ((SSE_R - SSE_C) / (k - g)) / (SSE_C / [n - (k + 1)])

F = (SSyy / k) / (SSE / [n - (k + 1)])

t_(n-(k+1)); α/2

R² = 1 - SSE / SSyy

Adjusted R²:
R²a = 1 - [(n - 1) / (n - (k + 1))] * (SSE / SSyy)
R²a = 1 - [(n - 1) / (n - (k + 1))] * (1 - R²)

Fk; n-(k+1); α

Fk-g; n-(k+1); α

MSE = SSE / [n - (k + 1)]

F = (R² / k) / ((1 - R²) / [n - (k + 1)])

MSE = SSE / [n - (k + 1)]

t_(n-(k+1)); α

F = (SS(model) / k) / (SSE / [n - (k + 1)])

R²(log(y)) = 1 - (Σ(yi - ŷi)² / Σ(yi - ȳ)²)

(VIF)i = 1 / (1 - Ri²)

Di = ((yi - ŷi)² / ((k + 1) * MSE)) * (hi / (1 - hi)²)

H = X(X'X)^(-1)X'

β̂ = (X'X)^(-1)X'y

SS(total) = y'y - n(ȳ)²

SSE = y'y - β̂'X'y

s² = SSE / [n - (k + 1)]

MS(model) = SS(model) / k

Confidence interval for βi:
βi ± t(n-(k+1); α/2) * s * √cii

t = βi / (s * √cii)

Confidence or prediction interval (matrix form):
ŷ ± t(n-(k+1); α/2) * √(s² * a'(X'X)^(-1)a)

ŷ ± t(n-(k+1); α/2) * √(s² * [1 + a'(X'X)^(-1)a])

εi = yi - ŷi

ε* = ε̂ - β̂j * xj

F = Larger MSE / Smaller MSE

A = (i - 0.375) / (n + 0.25)

E(εi) ≈ √MSE * Z(A)

h̄ = (k + 1) / n

hi ≥ 2(k + 1) / n

di = yi - ŷi

F(k+1); n-(k+1); α

DW = Σ(t=2 to n)(εt - ε(t-1))² / Σ(t=1 to n)εt²

d*i = di / s(di)

Confidence interval for predicted x:
x̂ ± t(n-2); α/2 * (s / b1) * √(1 + 1/n + (x̂ - x̄)² / SSxx)

x̂ = (yp - b0) / b1

x̄ = Σx / n

SSxx = Σx² - n(x̄)²

s = √MSE

D = [(t(n-2); α/2 * s / b1)²] * (1 / SSxx)

WSSE = Σ(wi * (yi - ŷi)²)

wi = 1 / σi²

wi = 1 / xi

wi = 1 / [ŷi * (1 - ŷi)]

E(y) = exp(β0 + β1x1 + ... + βkxk) / [1 + exp(β0 + β1x1 + ... + βkxk)]

π* = ln(π / (1 - π))

R²(prediction) = 1 - [Σ(i=1 to n1+n2)(yi - ŷi)²] / [Σ(i=n1+1 to n1+n2)(yi - ȳ)²]

MSE(prediction) = [Σ(i=n1+1 to n1+n2)(yi - ŷi)²] / [n2 - (k + 1)]

R²(Jackknife) = Σ(yi - ŷ(i))² / Σ(yi - ȳ)²

MSE(Jackknife) = Σ(yi - ŷ(i))² / [n - (k + 1)]

Studentized Residual = εi / [s * √(1 - hii)]

y* = β1*x1 + β2*x2 + ... + βk*xk + ε

β̂R = (X'*X + cI)^(-1)X'*y*

SSE = y'*y - β̂R'*X'*y

ŷi = (1 / √(n - 1)) * ((yi - ȳ) / sy)

x̂i = (1 / √(n - 1)) * ((xi - x̄) / sxi)

βi,R = (sy / sxi) * β̂i,R

β0,R = ȳ - β̂1,R * x̄1 - ... - β̂k,R * x̄k

Var(β̂R) = s² * (X'X + cI)^(-1) * X'X * (X'X + cI)^(-1)
