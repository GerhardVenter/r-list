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

#----------------------------------------------------------------------------

General Commands:

* Clear environment variables: rm(list = ls())

* Clear plots in RStudio:
  if (dev.cur() > 1) dev.off()  # close current plot window
  while (!is.null(dev.list())) dev.off()  # close all plots

#----------------------------------------------------------------------------

STAS202 FORMULA SHEET

#----------------------------------------------------------------------------

CHAPTER: 3 - Simple Linear Regression (SLR)
-------------------------------------------
Topic: Model and Parameter Estimation

	Formula: y = β0 + β1X1 + ε
→ Population model. Used to express the linear relationship between the independent variable (x) and dependent variable (y).

	y = β0 + β1X1 + β2X2 + ... + βkXk + ε
-------------------------------------------
	Formula: b1 = SSxy / SSxx and b0 = ȳ - b1x̄
→ Sample estimates for slope (b1) and intercept (b0).
→ Found using the least squares method.

	b1 = SSxy / SSxx
	b0 = ȳ - b1x̄
-------------------------------------------
Formulas for sums of squares:
→ Used to compute slope, intercept, and correlation.
→ Foundation of all regression calculations.

	SSxy = Σ(xi - x̄)(yi - ȳ) = Σ(xi * yi) - n * x̄ * ȳ

	SSxx = Σ(xi - x̄)² = Σ(xi²) - n(x̄)²

	SSyy = Σ(yi - ȳ)² = Σ(yi²) - n(ȳ)²

	SSyy = SSE / (1 - R²)
→ Rearranged identity linking explained and unexplained variation.
-------------------------------------------
	Formula: SSE = Σ(yi - ŷi)² = SSyy - b1SSxy
→ Residual sum of squares (unexplained variation).
→ Used to measure model fit.

	SSE = Σ(yi - ŷi)² = SSyy - b1SSxy
-------------------------------------------
Topic: Model Fit and Variability

Formulas:

	s² = SSE / (n - 2)
	σ(b1) = σ / √SSxx

→ Estimate of error variance and standard error of the slope.

	s² = SSE / (n - 2)
	σ(b1) = σ / √SSxx

Formula: SSyy = SSE / (1 - R²)
→ Rearranged identity linking explained and unexplained variation.
-------------------------------------------
Topic: Hypothesis Testing for Slope and Correlation
t-statistics for slope:
	t = b1 / s(b1) = b1 / (s / √SSxx)
→ Used for testing H₀: β1 = 0 (no linear relationship).

Confidence interval for b1:
Confidence interval for slope:
	b1 ± t(α/2) * s(b1)
→ Gives plausible range for true β1.

Correlation coefficient:
	r = SSxy / √(SSxx * SSyy)
→ Measures strength/direction of linear relationship.

t-test for correlation significance:
	t = r√(n - 2) / √(1 - r²)
→ Used to test if correlation differs from zero.

Coefficient of determination:
	r² = (SSyy - SSE) / SSyy = 1 - SSE / SSyy
→ Proportion of variation in y explained by x.
-------------------------------------------

Topic: Confidence and Prediction Intervals

Mean response (confidence interval):
	σ(ŷ) = σ * √(1/n + (xp - x̄)² / SSxx)
→ Standard error of mean response at xp.

Individual response (prediction interval):
	σ(y - ŷ) = σ * √(1 + 1/n + (xp - x̄)² / SSxx)
→ Standard error of a single predicted y-value.

#----------------------------------------------------------------------------

Key R commands you would use:

fit <- lm(y ~ x)
summary(fit)          # gives b0, b1, s, t, R²
confint(fit)          # confidence intervals
predict(fit, interval="confidence")
predict(fit, interval="prediction")
cor(x, y)             # correlation

#----------------------------------------------------------------------------

Topic: Confidence and Prediction Intervals

These formulas describe the uncertainty in predictions from a simple linear regression model.

Confidence interval for mean response:
	ŷ ± t(α/2) * s * √(1/n + (xp - x̄)² / SSxx)
→ Gives the range in which the average response at xp is expected to fall.
→ Used when estimating the mean of y for a specific x-value.

Prediction interval for individual response:
	ŷ ± t(α/2) * s * √(1 + 1/n + (xp - x̄)² / SSxx)
→ Gives the range for a new individual observation.
→ Wider than the confidence interval because it includes individual error.

-------------------------------------------

Topic: Alternative Computational Form (Simplified Regression Through Origin)

The next group of formulas assumes the model passes through the origin (no intercept, or transformed data).

Slope estimate:
	b1 = (Σxi * yi) / (Σxi²)
→ Slope formula when the regression line is forced through the origin.

Error variance estimate:
	s² = SSE / (n - 1)
→ Uses n - 1 degrees of freedom (since no intercept).

Residual sum of squares:
	SSE = Σyi² - b1Σxi * yi
→ Derived from least squares minimization with no intercept term.

-------------------------------------------

Topic: Standard Errors and t-tests

These measure how uncertain your estimates are.

Standard error of slope:
	s(b1) = s / √(Σxi²)
→ Used to test whether slope differs significantly from zero.

Test statistic for slope:
	t = (b1 - 0) / s(b1) = b1 / (s / √(Σxi²))
→ Tests H₀: β1 = 0 (no relationship).

Confidence interval for slope:
	b1 ± t(α/2) * s(b1)
→ Gives range of plausible values for true slope β1.

s(b1) = s / √(Σxi²)

t = (b1 - 0) / s(b1) = b1 / (s / √(Σxi²))

Confidence interval for slope:
b1 ± t(α/2) * s(b1) = b1 ± t(α/2) * (s / √(Σxi²))

-------------------------------------------

Topic: Precision of Predictions (through-origin case)

These versions are specialized for centered or origin-based data (Σx = 0).

Standard error of mean prediction:
	s(ŷ) = s * (xp / √(Σxi²))

Standard error of individual prediction:
	s(y - ŷ) = s * √(1 + xp² / Σxi²)

Confidence interval for mean response (origin model):
	ŷ ± t(α/2) * s(ŷ) = ŷ ± t(α/2) * s * (xp / √(Σxi²))

Prediction interval for individual response (origin model):
	ŷ ± t(α/2) * s(y - ŷ) = ŷ ± t(α/2) * s * √(1 + xp² / Σxi²)

#----------------------------------------------------------------------------

Exam R relevance

All of these are automatically handled in R with:

fit <- lm(y ~ x)
predict(fit, newdata=data.frame(x=xp), interval="confidence")
predict(fit, newdata=data.frame(x=xp), interval="prediction")

#----------------------------------------------------------------------------

Chapter: 4–5 — Multiple Linear Regression (MLR)
-----------------------------------------------
Topic: Significance Tests for Regression Coefficients

	Formula: t = bi / s(bi)
→ Tests if a specific slope coefficient (βi) differs significantly from zero.
→ Null hypothesis: H₀: βi = 0.
→ Used to identify which predictors are statistically significant.

R equivalent:

summary(fit)  # look at t value and Pr(>|t|)

-------------------------------------------

Topic: Partial F-test (Reduced vs. Full Model)

	Formula: F = ((SSE_R - SSE_C) / (k - g)) / (SSE_C / [n - (k + 1)])
→ Tests whether adding k - g new predictors significantly improves model fit.
→ H₀: Added coefficients = 0 (no improvement).
→ Used when comparing a reduced model (fewer predictors) to a complete (full) model.

R equivalent:

anova(reduced_model, full_model)

-------------------------------------------

Topic: Overall Model Significance (Global F-test)

	Formula: F = (SSyy / k) / (SSE / [n - (k + 1)])
→ Tests if the entire regression model explains a significant portion of variation in y.
→ H₀: β1 = β2 = ... = βk = 0.
→ Used to test the usefulness of the full model as a whole.

R equivalent:

summary(fit)$fstatistic

-------------------------------------------

Topic: Critical t-value notation

	Formula: t_(n-(k+1)); α/2
→ Symbol for the t critical value from the t-distribution with n - (k + 1) degrees of freedom.
→ Used for constructing confidence intervals for βi or predictions.

R equivalent:

qt(1 - α/2, df = n - (k + 1))

-------------------------------------------

Topic: Coefficient of Determination (R²)

	Formula: R² = 1 - SSE / SSyy
→ Proportion of total variation in y explained by all predictors together.
→ Used to evaluate overall goodness of fit.

R equivalent:

summary(fit)$r.squared

-------------------------------------------

Topic: Adjusted R² (Penalty for Extra Predictors)

Formulas:

	R²a = 1 - [(n - 1) / (n - (k + 1))] * (SSE / SSyy)
	R²a = 1 - [(n - 1) / (n - (k + 1))] * (1 - R²)


→ Adjusted R² corrects R² for the number of predictors used.
→ Used to compare models with different numbers of predictors.

R equivalent:

summary(fit)$adj.r.squared

#----------------------------------------------------------------------------

Topic: Critical Values for F and t Tests

Formulas:

	Fk; n-(k+1); α  
	Fk-g; n-(k+1); α  
	t_(n-(k+1)); α


→ These are critical values from the F and t distributions.
→ Used to determine rejection regions for hypothesis tests.
→
• Fk; n-(k+1); α → overall F-test for full model (numerator df = k).
• Fk-g; n-(k+1); α → partial F-test comparing reduced (g) and full (k) models.
• t_(n-(k+1)); α → critical t-value for confidence intervals or t-tests.

R equivalent:

qf(1 - α, df1 = k, df2 = n - (k + 1))
qt(1 - α/2, df = n - (k + 1))

-------------------------------------------

Topic: Mean Square Error (MSE)

Formula:
	MSE = SSE / [n - (k + 1)]
→ Estimate of the residual variance (σ²).
→ Used in F-tests and for standard errors of coefficients.

R equivalent:

summary(fit)$sigma^2
deviance(fit) / df.residual(fit)

-------------------------------------------

Topic: Overall Model Significance (F-statistic)

Formula:
	F = (R² / k) / ((1 - R²) / [n - (k + 1)])
→ Tests whether at least one predictor has a nonzero slope.
→ Equivalent to comparing model with no predictors vs. full model.
→ Numerator df = k, denominator df = n - (k + 1).

R equivalent:

summary(fit)$fstatistic


Alternate Form (ANOVA definition):
	F = (SS(model) / k) / (SSE / [n - (k + 1)])
→ Equivalent expression using model sums of squares directly.
→ Used in ANOVA tables for regression.

R equivalent:

anova(fit)

-------------------------------------------

Topic: Model Evaluation after Transformation

Formula:
	R²(log(y)) = 1 - (Σ(yi - ŷi)² / Σ(yi - ȳ)²)
→ R² calculated for a log-transformed dependent variable.
→ Used when model is fitted to log(y) rather than y itself (e.g., exponential models).
→ Measures proportion of variability in log(y) explained by predictors.

R equivalent:

summary(lm(log(y) ~ x1 + x2))$r.squared

#----------------------------------------------------------------------------

Chapter: 9 — Model Assessment and Diagnostics

(Also partly Chapter 4–5 for the matrix representation of MLR)

#----------------------------------------------------------------------------

Topic: Collinearity Diagnostics

Formula:
	(VIF)i = 1 / (1 - Ri²)
→ Variance Inflation Factor for predictor i.
→ Detects multicollinearity (linear dependence among predictors).
→ High VIF (> 10) suggests that xi is highly correlated with other predictors.
→ Ri² is the R² value when regressing xi on all other predictors.

R equivalent:

car::vif(fit)

-------------------------------------------

Topic: Influence Diagnostics

Formula:
	Di = ((yi - ŷi)² / ((k + 1) * MSE)) * (hi / (1 - hi)²)
→ Cook’s Distance — measures influence of observation i on fitted coefficients.
→ Combines residual size and leverage hi.
→ Large Di (> 1 or high relative to others) indicates influential outlier.

R equivalent:

cooks.distance(fit)

-------------------------------------------

Topic: Leverage (Hat Matrix)

Formula:
	H = X(X'X)^(-1)X'
→ Hat matrix, maps y-values to fitted values ŷ = Hy.
→ Diagonal elements hi measure leverage — how far xi is from the mean of x.
→ Used in Cook’s distance, standardized residuals, and diagnostic plots.

R equivalent:

hatvalues(fit)

-------------------------------------------

Topic: Matrix Form of Least Squares Estimation

Formula:
	β̂ = (X'X)^(-1)X'y
→ Vectorized form for estimating regression coefficients.
→ Fundamental to all linear models, used in deriving SSE, Var(β̂), and tests.

R equivalent:
Automatically computed by lm(); you can check it manually:

solve(t(X) %*% X) %*% t(X) %*% y

-------------------------------------------

Topic: Model Sums of Squares in Matrix Form

Formulas:

	SS(total) = y'y - n(ȳ)²  
	SSE = y'y - β̂'X'y  
	s² = SSE / [n - (k + 1)]

→ Express regression variability in matrix terms.
→ SSE measures unexplained variation, s² estimates error variance σ².
→ Used to derive F- and t-tests in MLR context.

R equivalent:

anova(fit)
summary(fit)$sigma^2

#----------------------------------------------------------------------------

Chapter: 5 — Inference in Multiple Linear Regression

(The last two also connect with Chapter 9 — residual analysis)

#----------------------------------------------------------------------------

Topic: ANOVA Component for Regression

Formula:
	MS(model) = SS(model) / k
→ Mean Square for Regression.
→ Measures average variation in y explained by the predictors.
→ Used in F-test for overall model significance:
F = MS(model) / MSE

R equivalent:

anova(fit)

-------------------------------------------

Topic: Confidence Intervals for Regression Coefficients

Formula:
Confidence interval for βi: 
	βi ± t(n-(k+1); α/2) * s * √cii
→ Estimates plausible range for true slope βi.
→ cii = ith diagonal element of (X'X)^(-1).
→ Used to determine whether each predictor’s effect is statistically significant.

Formula:
	t = βi / (s * √cii)
→ Corresponding t-test for H₀: βi = 0.
→ Equivalent to output in the “Coefficients” table in R.

R equivalent:

summary(fit)$coefficients
confint(fit)

-------------------------------------------

Topic: Confidence and Prediction Intervals (Matrix Form)

These are generalized forms of the standard SLR intervals — valid for any multiple regression model.

Confidence interval for mean prediction:
	ŷ ± t(n-(k+1); α/2) * √(s² * a'(X'X)^(-1)a)
→ Gives confidence bounds for the mean response at specific predictor values.
→ a is the vector of predictor values (including 1 for the intercept).

Prediction interval for individual response:
	ŷ ± t(n-(k+1); α/2) * √(s² * [1 + a'(X'X)^(-1)a])
→ Adds “+1” to account for extra uncertainty in predicting an individual observation.

R equivalent:

predict(fit, newdata, interval="confidence")
predict(fit, newdata, interval="prediction")

-------------------------------------------

Topic: Residuals and Adjusted Residuals

Formula:
	εi = yi - ŷi
→ Regular residual (difference between observed and fitted value).
→ Used to check assumptions (normality, constant variance, independence).

Formula:
	ε* = ε̂ - β̂j * xj
→ Residual with a variable’s effect removed.
→ Used in partial regression or added-variable plots to visualize each predictor’s unique contribution after adjusting for others.

R equivalent:

residuals(fit)
rstudent(fit)

#----------------------------------------------------------------------------

Chapter: 9 — Model Assessment and Diagnostics

(All deal with assumption checking, residuals, and influence.)

#----------------------------------------------------------------------------

Topic: Checking Equality of Variances (Model Comparison)

Formula:
	F = Larger MSE / Smaller MSE
→ Used to test whether two regression models (or two data sets) have equal error variances.
→ H₀: σ₁² = σ₂².
→ Often used before pooling models or testing structural differences.

R equivalent:

var.test(resid(model1), resid(model2))

-------------------------------------------

Topic: Normal Probability Plot of Residuals

Formula:
	A = (i - 0.375) / (n + 0.25)
→ Calculates the expected percentile position for the i-th ordered residual.
→ Used to approximate expected z-scores for normal Q-Q plots.

Formula:
	E(εi) ≈ √MSE * Z(A)
→ Expected residual value under normality assumption.
→ Used to compare actual residuals to theoretical normal residuals in Q-Q plots.

R equivalent:

qqnorm(resid(fit))
qqline(resid(fit))
-------------------------------------------

Topic: Leverage and Influence Rules

Formula:
	h̄ = (k + 1) / n
→ Mean leverage across all observations.

Formula:
	hi ≥ 2(k + 1) / n
→ Rule of thumb for identifying high-leverage points.
→ Observations exceeding this threshold may strongly influence the fitted model.

R equivalent:

hatvalues(fit)
which(hatvalues(fit) > 2 * (k + 1) / n)

-------------------------------------------

di = yi - ŷi

-------------------------------------------

Topic: Critical F Value for Overall Significance

Formula:
	F(k+1); n-(k+1); α
→ Critical value from the F-distribution for model significance tests.
→ Used for overall F-test in multiple regression.

R equivalent:

qf(1 - α, df1 = k + 1, df2 = n - (k + 1))

#----------------------------------------------------------------------------

Chapters Involved:

Chapter 3 — Simple Linear Regression (SLR) → for inverse prediction and interval estimation.

Chapter 9 — Diagnostics → for Durbin–Watson and standardized residuals.

#----------------------------------------------------------------------------

Topic (Ch. 9): Detecting Autocorrelation

Formula:
	DW = Σ(t=2 to n)(εt - ε(t-1))² / Σ(t=1 to n)εt²
→ Durbin–Watson statistic used to test for first-order autocorrelation in residuals.
→ Values near 2 indicate independence; below 2 suggests positive autocorrelation.
→ Applies when data are time-ordered.

R equivalent:

lmtest::dwtest(fit)

-------------------------------------------

Topic (Ch. 9): Standardized Residuals

Formula:
	d*i = di / s(di)
→ Standardized (or studentized) residual for observation i.
→ Used to detect outliers; values beyond ±2 or ±3 are concerning.

R equivalent:

rstandard(fit)
rstudent(fit)

-------------------------------------------

Topic (Ch. 3): Confidence Interval for Predicted x

Formula:
	x̂ ± t(n-2); α/2 * (s / b1) * √(1 + 1/n + (x̂ - x̄)² / SSxx)
→ Provides uncertainty bounds for the predicted x from inverse regression.
→ Wider than a standard prediction interval due to inversion of regression equation.

-------------------------------------------

Topic (Ch. 3): Inverse Prediction (Estimating x from y)

Formula:
	x̂ = (yp - b0) / b1
→ Used to predict an x-value that corresponds to a given y (yp).
→ Common in calibration and standard curve problems.

-------------------------------------------

Topic (Ch. 3): Supporting Computations for SLR

Formulas:

	x̄ = Σx / n
	SSxx = Σx² - n(x̄)²
	s = √MSE


→ Basic definitions for computing slope, intercept, and variability in regression.
→ Used to evaluate slope precision and construct intervals.

-------------------------------------------

Topic (Ch. 3): Margin of Error Component

Formula:
	D = [(t(n-2); α/2 * s / b1)²] * (1 / SSxx)
→ Term used inside the confidence interval for predicted x.
→ Represents the uncertainty propagation through the slope.

#----------------------------------------------------------------------------

Chapter: 8 — Weighted and Nonlinear Models (Weighted Least Squares and Logistic Regression)**

#----------------------------------------------------------------------------

Topic: Weighted Least Squares (WLS)

Used when the assumption of constant variance (homoscedasticity) is violated — i.e., the variability of residuals changes with x.

Formula:
	WSSE = Σ(wi * (yi - ŷi)²)
→ Weighted Sum of Squares for Error.
→ The objective minimized in a weighted least squares fit.
→ Each residual is weighted by wi to stabilize variance.

Formulas for weights (wi):

	wi = 1 / σi²
	wi = 1 / xi
	wi = 1 / [ŷi * (1 - ŷi)]


→ Weighting rules depending on the form of heteroscedasticity.
→ Examples:
• 1 / σi² → general case (known variances).
• 1 / xi → when variance ∝ xi.
• 1 / [ŷi(1 - ŷi)] → used in logistic regression (variance of Bernoulli data).

R equivalents:

lm(y ~ x, weights = w)


For logistic regression:

glm(y ~ x, family = binomial)

-------------------------------------------

Topic: Logistic Regression Model

Used when the dependent variable y is binary (0/1).

Formula:
	E(y) = exp(β0 + β1x1 + ... + βkxk) / [1 + exp(β0 + β1x1 + ... + βkxk)]
→ Logistic function.
→ Models probability that y = 1.
→ Ensures output is between 0 and 1.

Formula:
	π* = ln(π / (1 - π))
→ Logit transformation.
→ Linearizes the logistic model so it can be fitted via linear predictors.
→ Corresponds to:
	π* = β0 + β1x1 + ... + βkxk

R equivalent:

glm(y ~ x1 + x2, family = binomial)

#----------------------------------------------------------------------------

Chapter: 9 — Model Validation and Diagnostics

(These measure predictive performance and check model reliability.)

#----------------------------------------------------------------------------

Topic: Prediction Assessment (Cross-Validation and Holdout)

Used to evaluate how well the model predicts unseen data, not just how well it fits the training data.

Formula:
	R²(prediction) = 1 - [Σ(i=1 to n1+n2)(yi - ŷi)²] / [Σ(i=n1+1 to n1+n2)(yi - ȳ)²]
→ Predictive R² (Holdout R²).
→ Measures how well the model trained on subset 1 predicts responses in subset 2.
→ Higher R²(prediction) = better predictive ability on new data.

Formula:
	MSE(prediction) = [Σ(i=n1+1 to n1+n2)(yi - ŷi)²] / [n2 - (k + 1)]
→ Mean Square Error of Prediction.
→ Quantifies average squared error when predicting new observations.
→ Used in model comparison or cross-validation exercises.

R equivalent:

mean((y_test - predict(fit, newdata=test_data))^2)

-------------------------------------------

Topic: Jackknife Cross-Validation

Used to test stability of model estimates and predictive reliability.

Formula:
	R²(Jackknife) = Σ(yi - ŷ(i))² / Σ(yi - ȳ)²
→ R² using leave-one-out predictions (ŷ(i) = fitted value when case i is excluded).
→ Lower values suggest sensitivity to specific data points.

Formula:
	MSE(Jackknife) = Σ(yi - ŷ(i))² / [n - (k + 1)]
→ Corresponding jackknife mean squared error.
→ Measures predictive variance while systematically omitting each observation.

R equivalent:

cv.lm(data, form.lm = fit)  # from DAAG package

-------------------------------------------

Topic: Studentized Residuals (Outlier Detection)

Formula:
	Studentized Residual = εi / [s * √(1 - hii)]
→ Residual standardized by its estimated standard deviation.
→ Used to identify outliers — typically beyond ±2 or ±3 are unusual.
→ Incorporates leverage (hii) to account for influence of each data point.

R equivalent:

rstudent(fit)

#----------------------------------------------------------------------------

Chapter: 9 — Model Diagnostics and Remedial Measures

(Specifically, Ridge Regression and Standardization for multicollinearity problems.)

#----------------------------------------------------------------------------

Topic: Ridge Regression (Shrinkage Estimation)

Ridge regression modifies the OLS estimation procedure to stabilize coefficient estimates when predictors are highly correlated (multicollinearity).

Formula:
	y* = β1*x1 + β2*x2 + ... + βk*xk + ε
→ Model written for standardized variables (mean 0, variance 1).
→ Ensures all predictors contribute on the same scale.

Formula:
	β̂R = (X'*X + cI)^(-1)X'*y*
→ Ridge estimator.
→ Adds constant c (ridge parameter > 0) to the diagonal of X′X to shrink coefficients toward 0.
→ Reduces variance at the cost of introducing slight bias.

Formula:
	SSE = y'*y - β̂R'*X'*y
→ Residual sum of squares for ridge regression.
→ Used to evaluate model fit under ridge shrinkage.

R equivalent:

ridge <- MASS::lm.ridge(y ~ x1 + x2 + ..., lambda = c)

-------------------------------------------

Topic: Standardization of Variables

Ridge regression typically operates on standardized data (zero mean, unit variance).
These transformations define that process.

Formulas:

	ŷi = (1 / √(n - 1)) * ((yi - ȳ) / sy)
	x̂i = (1 / √(n - 1)) * ((xi - x̄) / sxi)


→ Convert raw data into standardized scale.
→ Prevents predictors with large scales from dominating the penalty term.

-------------------------------------------

Topic: Converting Back to Original Scale

After fitting ridge regression on standardized variables, you need to transform coefficients back to the original data scale.

Formulas:

	βi,R = (sy / sxi) * β̂i,R
	β0,R = ȳ - β̂1,R * x̄1 - ... - β̂k,R * x̄k


→ Adjust ridge coefficients and intercept back to unstandardized units.
→ Allows predictions on the original data scale.

-------------------------------------------

Topic: Variance of Ridge Coefficients

Formula:
	Var(β̂R) = s² * (X'X + cI)^(-1) * X'X * (X'X + cI)^(-1)
→ Describes how ridge regularization affects estimator variance.
→ As c increases, variance decreases (more stable coefficients).
→ Useful for comparing ridge vs. OLS precision.

#----------------------------------------------------------------------------

Durbin–Watson

#----------------------------------------------------------------------------

What it is
Durbin–Watson (DW) tests for first-order autocorrelation in regression residuals.
Used when data are time-ordered (or ordered sequences), not for randomly ordered cross-sectional data.

Formula
DW = Σ(t=2 to n) (e_t - e_{t-1})² / Σ(t=1 to n) e_t²
where e_t are residuals from your fitted model.

Hypotheses
H0: residuals are independent (no first-order autocorrelation).
H1: residuals are autocorrelated (often specifically positive autocorrelation).

Rough interpretation (exam level)
Always: 0 ≤ DW ≤ 4.

DW ≈ 2 → no autocorrelation.

DW < 2 → positive autocorrelation.

Strong evidence if DW is much less than 2 (e.g. < 1.2).

DW > 2 → negative autocorrelation.

Strong evidence if DW is much greater than 2 (e.g. > 2.8).

More precise: DW ≈ 2(1 - r̂), where r̂ is lag-1 sample autocorrelation of residuals.

When to use / when examiner cares

Time series style data: observations in sequence (days, months, runs, positions).

Typical exam setup: “Fit regression; check independence assumption using Durbin–Watson.”

If independence fails:

Standard OLS t and F tests are unreliable.

You mention “consider time-series methods or add correlation structure” (that’s enough for your course).

What to write in an exam
Example style:

“DW = 1.05 (< 2), indicating positive autocorrelation; independence of errors is questionable.”

“DW = 2.03 (very close to 2), no evidence of autocorrelation; independence assumption is reasonable.”

“DW = 3.10 (> 2), suggests negative autocorrelation.”

If they give critical bounds (dL, dU), you compare DW to those. If not, the rule-of-thumb above is acceptable.

R code you should know

Basic:

fit <- lm(y ~ x1 + x2)      # your regression
e   <- resid(fit)

# Manual DW:
DW <- sum((e[-1] - e[-length(e)])^2) / sum(e^2)
DW


Using lmtest (if allowed):

install.packages("lmtest")  # once
library(lmtest)
dwtest(fit)


dwtest(fit) outputs:

DW statistic

p-value (test H0: no autocorrelation vs H1: positive autocorrelation by default)

Interpretation in one line:

If p-value < 0.05 → evidence of autocorrelation → independence violated.

If p-value ≥ 0.05 → no evidence against independence.

Link to your formula sheet
Your sheet’s line:
DW = Σ(ε_t - ε_{t-1})² / Σε_t²
belongs to:

Chapter 9: “Checking error independence (Durbin–Watson).”
Use only when:

Data have natural order.

You are explicitly asked about independence of residuals.

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
1. Variance Inflation Factor (VIF)
#----------------------------------------------------------------------------

1. Variance Inflation Factor (VIF)

Purpose:
VIF detects multicollinearity — when predictors are highly correlated with each other.
This causes unstable slope estimates and inflated standard errors.

Formula: VIFᵢ = 1 / (1 − Rᵢ²)
where Rᵢ² is the R² from regressing xᵢ on all other x’s.

Interpretation:

VIF = 1 → xᵢ independent of others.

1 < VIF < 5 → moderate correlation, usually acceptable.

VIF > 10 → serious multicollinearity; coefficients unstable.

Consequence: high VIF ⇒ (High VIF inflates) / large standard errors ⇒ t-tests unreliable.

How to calculate manually (no external packages):
Example for model y ~ x1 + x2 + x3
fit1 <- lm(x1 ~ x2 + x3)
fit2 <- lm(x2 ~ x1 + x3)
fit3 <- lm(x3 ~ x1 + x2)
Then get each R² from summary(fit)$r.squared
VIF1 <- 1 / (1 - summary(fit1)$r.squared)
VIF2 <- 1 / (1 - summary(fit2)$r.squared)
VIF3 <- 1 / (1 - summary(fit3)$r.squared)

Example:
x1 <- 1:10
x2 <- x1 + rnorm(10, 0, 0.5)
y <- 3 + 2*x1 + rnorm(10)
fit <- lm(y ~ x1 + x2)
fit_x1 <- lm(x1 ~ x2)
fit_x2 <- lm(x2 ~ x1)
VIF_x1 <- 1 / (1 - summary(fit_x1)$r.squared)
VIF_x2 <- 1 / (1 - summary(fit_x2)$r.squared)
print(c(VIF_x1, VIF_x2))

Output example: both ≈ 15 → strong multicollinearity.

Fixes:
Remove or combine correlated predictors, or use ridge regression.

R code:

install.packages("car")

library(car)
vif(fit)

#----------------------------------------------------------------------------
2. Cook’s Distance (Dᵢ)
#----------------------------------------------------------------------------

Purpose:
Measures influence of each observation on the fitted regression coefficients.

Formula: Dᵢ = [(eᵢ²)/( (k + 1) MSE )] × [hᵢ / (1 − hᵢ)²]
where ei = residual, hi = leverage.

Interpretation:

Dᵢ < 0.5 → no influence. (Not influential)

0.5 ≤ Dᵢ < 1 → watch. (Check the observation)

Dᵢ ≥ 1 → strong influence; investigate or refit without that case. (Influential point, possibly distorting model)

R Example (base R only):
fit <- lm(mpg ~ wt + hp, data = mtcars)
cd <- cooks.distance(fit)
plot(cd, type="h", ylab="Cook's D")
abline(h = 1, col="red", lty=2)
which(cd > 1)

Interpretation:
Observations with D > 1 are influential. Check their effect by refitting without them.

R code:

cooks.distance(fit)
plot(cooks.distance(fit), type="h")

#----------------------------------------------------------------------------
3. Leverage (hᵢ)
#----------------------------------------------------------------------------

Purpose: identify observations with unusual x-values (high leverage).
Identifies observations with unusual X-values (extreme predictor combinations).

Formula: hᵢ = xᵢ′ (X′X)⁻¹ xᵢ (diagonal of H = X(X′X)⁻¹X′)

Mean leverage: h̄ = (k + 1)/n

Interpretation:

hᵢ ≈ h̄ → typical. (hi < 2 * mean leverage → Normal)

hᵢ > 2h̄ → high leverage point (potentially influential).
(hi > 2 * mean leverage → High leverage (could have large impact)
High leverage + large residual → Influential.)

High leverage + large residual ⇒ influential.

R Example (base R):
fit <- lm(mpg ~ wt + hp, data = mtcars)
h <- hatvalues(fit)
mean_h <- (length(coef(fit))) / nrow(mtcars)
which(h > 2 * mean_h)
plot(h, type="h", ylab="Leverage")
abline(h = 2 * mean_h, col="red", lty=2)

Interpretation:
Points with large hi values have unusual X combinations.

R code:

hatvalues(fit)
which(hatvalues(fit) > 2 * (k + 1) / n)

#----------------------------------------------------------------------------
4. Studentized Residual (rᵢ)
#----------------------------------------------------------------------------

Purpose: detect outliers in y (unusual residuals after accounting for leverage).
(Detects outliers in Y after accounting for leverage.)

Formula: rᵢ = eᵢ / [s √(1 − hᵢ)]
where s = standard error of estimate, hi = leverage.

Interpretation:

|rᵢ| ≤ 2 → typical.

|rᵢ| ≈ 2–3 → suspect.

|rᵢ| > 3 → outlier (check data or model).

(|ri| < 2 → Normal residual
|ri| 2–3 → Moderately large
|ri| > 3 → Outlier candidate)

R Example (base R):
fit <- lm(mpg ~ wt + hp, data = mtcars)
r <- rstudent(fit)
which(abs(r) > 3)
plot(r, ylab="Studentized Residuals")
abline(h = c(-3,3), col="red", lty=2)

Interpretation:
Points with |ri| > 3 are potential outliers and should be investigated.

R code:

rstudent(fit)
plot(rstudent(fit))

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

Hypotheses Tests

#----------------------------------------------------------------------------

1. t-Test for a Single Slope (Simple Linear Regression)

Purpose: test if the slope differs from 0.

H₀ : β₁ = 0
H₁ : β₁ ≠ 0

Statistic: t = b₁ / s(b₁) with df = n − 2
Reject H₀ if |t| > t₍α / 2, n − 2₎.

R: summary(fit) → check “Pr(>|t|)”.

-------------------------------------------

2. Overall F-Test (Model Significance)

Purpose: test if any predictor is useful.

H₀ : β₁ = β₂ = … = βₖ = 0
H₁ : at least one βᵢ ≠ 0

Statistic: F = MSR / MSE = (R² / k) / ((1 − R²)/(n − (k + 1)))
Reject H₀ if F > F₍k, n − (k + 1), α₎.

R: anova(fit) or summary(fit).

-------------------------------------------

3. Partial F-Test (Comparing Models)

Purpose: test if adding variables improves fit.

H₀ : β(g+1)…βk = 0
H₁ : at least one added β ≠ 0

Statistic: F = ((SSE_R − SSE_F)/g) / (SSE_F / (n − (k + 1)))
Reject H₀ if F > F₍g, n − (k + 1), α₎.

R:

anova(reduced, full)

-------------------------------------------

4. t-Tests for Individual Coefficients (Multiple Regression)

Purpose: check significance of each βᵢ.

H₀ : βᵢ = 0
H₁ : βᵢ ≠ 0

Statistic: t = bᵢ / s(bᵢ) (df = n − (k + 1))

R: summary(fit).

-------------------------------------------

5. Piecewise Regression Tests

Used when slope/intercept changes at x = k.

- Slope Change: H₀ : β₂ = 0 (use t for β₂)

- Discontinuity: H₀ : β₃ = 0 (use t for β₃)

- Joint Slope + Level: H₀ : β₂ = β₃ = 0 (use partial F with 2 df)

R: coefficients via summary(fit) or anova(reduced, full).

-------------------------------------------

6. Tests for Model Comparison or Transformations

Used to compare alternate fits.

- Equal Variance Test: F = Larger MSE / Smaller MSE H₀ : σ₁² = σ₂²

- Log or Power Transforms: compare R² (log y vs y).

R: var.test(resid(model1), resid(model2)).

-------------------------------------------

7. Durbin–Watson Test (Error Independence)

H₀ : errors independent
H₁ : positive autocorrelation

Statistic: DW = Σ(e_t − e_{t−1})² / Σ e_t²
DW ≈ 2 → no autocorr; DW < 2 → positive; DW > 2 → negative.

R:

library(lmtest)
dwtest(fit)

-------------------------------------------

8. Outlier / Influence Tests

Not strict hypothesis tests but used diagnostically.

Studentized Residual: t-test for outlier
 H₀ : observation fits model
 Reject if |t| > 2 or 3.
 rstudent(fit)

Cook’s Distance (Dᵢ): Influence > 1 often flagged.
 cooks.distance(fit)

-------------------------------------------

9. Logistic Regression (if in scope)

H₀ : βᵢ = 0 (no effect on log-odds)
Statistic: z = β̂ᵢ / SE(β̂ᵢ)

R: summary(glm(family=binomial))

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

1. R (Correlation coefficient)

Measures linear strength and direction between x and y.

Range −1 ≤ R ≤ 1.

|R| close to 1 → strong linear relationship.

R > 0 → positive slope, R < 0 → negative slope.

In simple regression, R = √R² with the sign of b₁.

Units: none.

-------------------------------------------

2. R² (Coefficient of Determination)

Proportion of variability in y explained by the regression model.

0 ≤ R² ≤ 1.

R² = 1 → perfect fit; R² = 0 → model explains nothing.

Interpretation: “The model explains R²×100 % of variation in y.”

Increases when you add predictors, even unhelpful ones.

-------------------------------------------

3. Adjusted R²

Penalizes adding irrelevant variables.

Formula: R²ₐ = 1 – (1 – R²)(n – 1)/(n – (k + 1)).

Compare models: larger R²ₐ → better balance of fit vs complexity.

-------------------------------------------

4. SSE (Error Sum of Squares)

SSE = Σ(yi – ŷi)².

Measures unexplained variation in y.

Smaller SSE → better fit.

Used in F- and t-tests.

Decreases as model becomes more flexible.

-------------------------------------------

5. SSR (Regression Sum of Squares)

SSR = Σ(ŷi – ȳ)².

Variation in y explained by the model.

SSR + SSE = SST (total sum of squares).

-------------------------------------------

6. MSE (Mean Square Error)

MSE = SSE / df = SSE / [n – (k + 1)].

Estimate of the error variance σ².

Used to compute s = √MSE.

Smaller MSE → better model.

-------------------------------------------

7. s (Standard Error of Estimate)

s = √MSE.

Average distance that observed y values fall from regression line.

Units: same as y.

Smaller s → points closer to fitted line.

-------------------------------------------

8. s² (Estimated Variance of Errors)

s² = MSE = σ̂².

Used in confidence and prediction intervals.

Represents variability of residuals.

-------------------------------------------

9. p-value

Probability of observing test statistic ≥ |Tobs| assuming H₀ true.

Small p (< α) → reject H₀ → evidence of significance.

Large p → insufficient evidence; coefficient may not matter.

Report as: “p = 0.012 < 0.05 → β₁ significant.”

-------------------------------------------

10. F-statistic

Tests overall model significance.

Large F → at least one βᵢ ≠ 0.

Compare to F₍k, n – (k + 1)₎ or use p-value.

-------------------------------------------

11. t-statistic

Tests individual slope or parameter.

t = bᵢ / s(bᵢ).

Large |t| → βᵢ significant.

Sign of t matches sign of estimated bᵢ.

-------------------------------------------

12. s(bᵢ) (Standard Error of Coefficient)

Measures uncertainty in bᵢ.

Smaller s(bᵢ) → more precise estimate.

Used to compute t and confidence intervals.

-------------------------------------------

13. Confidence Interval for βᵢ

βᵢ ± t₍α / 2₎ × s(bᵢ).

Range of plausible true slope values.

If 0 not inside → βᵢ significant at α.

-------------------------------------------

14. Confidence Interval for Mean Response

ŷ ± t × s × √(1/n + (x₀ – x̄)² / SSxx).

Predicts expected mean y for given x₀.

Narrower than prediction interval.

-------------------------------------------

15. Prediction Interval for Individual Response

ŷ ± t × s × √(1 + 1/n + (x₀ – x̄)² / SSxx).

Predicts future single y.

Always wider (includes random error).

-------------------------------------------

16. VIF (Variance Inflation Factor)

VIFᵢ = 1 / (1 – Rᵢ²).

Detects multicollinearity.

VIF > 10 → serious collinearity problem.

-------------------------------------------

17. Cook’s Distance (Dᵢ)

Influence measure; large Dᵢ (> 1) → observation i has strong effect on coefficients.

-------------------------------------------

18. hᵢ (Leverage)

Diagonal of H = X(X′X)⁻¹X′.

h̄ = (k + 1)/n.

hᵢ > 2h̄ → high leverage.

-------------------------------------------

19. Studentized Residual

eᵢ / (s√(1 – hᵢ)).

|value| > 2 or 3 → possible outlier.

-------------------------------------------

20. DW (Durbin–Watson)

Tests autocorrelation (see previous summary).

≈ 2 → independent errors; < 2 → positive correlation.

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

INTERPRETATION AND EXAM STRATEGY GUIDE

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

1. INTERPRETING MODEL COEFFICIENTS

#----------------------------------------------------------------------------

Simple Linear Regression (y ~ x):

b0 (intercept): predicted value of y when x = 0.
If x = 0 is not meaningful, say “not practically interpretable.”

b1 (slope): estimated change in y for a one-unit increase in x.
Example: “For every 1 hour increase in exposure, the number of surviving cells increases by 0.15.”

Multiple Linear Regression (y ~ x1 + x2):

bi: change in y for a one-unit increase in xi, holding all other variables constant.
Example: “When temperature is fixed, an extra gram of fertilizer increases growth by 0.8 units.”

#----------------------------------------------------------------------------

2. INTERPRETING R AND R^2

#----------------------------------------------------------------------------

R^2 = proportion of variability in y explained by the model.

- R^2 = 0.83 → “83% of variation in y is explained by the model.”

- Low R^2 does not mean the model is useless; the slope can still be significant.

- Compare R^2 only for models using the same dependent variable.

#----------------------------------------------------------------------------

3. ADJUSTED R^2

#----------------------------------------------------------------------------

Use when comparing models with different numbers of predictors.
If adjusted R^2 decreases after adding a variable, the variable does not improve prediction.

#----------------------------------------------------------------------------

4. INTERPRETING P-VALUES

#----------------------------------------------------------------------------

When p < 0.05 → reject H0 → “Evidence that variable x has a significant linear relationship with y.”
When p > 0.05 → insufficient evidence of relationship.

Example statement:
“The estimated slope for x2 is 0.42 (p = 0.03). This means that, holding other variables constant, a one-unit increase in x2 increases y by 0.42, and the effect is statistically significant at alpha = 0.05.”

#----------------------------------------------------------------------------

5. CONFIDENCE INTERVAL INTERPRETATION

#----------------------------------------------------------------------------

For a slope (bi):
“We are 95% confident that the true slope lies between 0.10 and 0.35.”
If the interval includes 0 → not significant.

For predictions:

Confidence interval = range for mean response.

Prediction interval = range for an individual future observation (always wider).

#----------------------------------------------------------------------------

6. COMMON DIAGNOSTIC INTERPRETATIONS

#----------------------------------------------------------------------------

Residual plot: random scatter → good; curved pattern → wrong model; funnel → unequal variance.
Normal Q-Q plot: points along line → normal; S-shape → skewed errors.
Cook’s D: D > 1 → influential point.
Leverage (hi): large hi > 2 * mean leverage → unusual x-values.
Studentized residual: |ri| > 3 → possible outlier.
VIF: > 10 → multicollinearity problem.
Durbin-Watson: around 2 → independent errors; below 2 → positive correlation.

Typical phrasing:
“Residuals show random scatter, indicating constant variance and a suitable linear model.”

#----------------------------------------------------------------------------

7. WHEN ASKED “IS THE MODEL ADEQUATE?”

#----------------------------------------------------------------------------

Answer checklist:

R^2 or adjusted R^2 high enough for practical prediction.

Key predictors have p < 0.05.

Residuals roughly normal and random.

No high Cook’s D or leverage points.

Durbin-Watson close to 2.

If all satisfied → “Yes, the model assumptions are met and the model is adequate.”

#----------------------------------------------------------------------------

8. PIECEWISE OR DISCONTINUOUS MODELS

#----------------------------------------------------------------------------

Before k: E(y) = b0 + b1x
After k: E(y) = b0 + b1x + b2(x - k) (+ b3 if discontinuous)

If b2 significant → slope changes after k.
If b3 significant → level jump at k.
If both → joint F-test shows slope + level change.

Example:
“At x = 70 hours, the slope changes significantly (p = 0.02), meaning growth rate increases after that point.”

#----------------------------------------------------------------------------

9. WEIGHTED AND LOGISTIC MODELS

#----------------------------------------------------------------------------

Weighted regression:
“Weights correct for unequal variance. Larger weights indicate more reliable observations.”

Logistic regression:
“Exp(b1) = 1.8 means that each one-unit increase in x raises the odds of success by 80%.”

#----------------------------------------------------------------------------

10. F-TEST AND PARTIAL F-TEST INTERPRETATION

#----------------------------------------------------------------------------

Large F → model or added variables improve fit.
Example: “F = 12.4, p = 0.001 → The full model explains significantly more variation than the reduced model.”
If p > 0.05 → “Added variable does not significantly improve the model.”

#----------------------------------------------------------------------------

11. INTERPRETATION PATTERNS TO MEMORIZE

#----------------------------------------------------------------------------

“Holding all else constant, …” → required phrase for multiple regression.

“Significant slope means a linear relationship exists.”

“If CI for slope includes 0, not significant.”

“If Cook’s D > 1, that point heavily influences the model.”

“If residual variance grows with x, constant variance assumption fails.”

#----------------------------------------------------------------------------

12. INTERPRETATION TEMPLATE FOR ANY QUESTION

#----------------------------------------------------------------------------

State what the statistic measures.

Give the observed value and threshold.

State what it implies about the model.

Mention whether assumptions are satisfied.

Example:
“R^2 = 0.84 shows that 84% of the variation in plant growth is explained by temperature and humidity. Both predictors are significant (p < 0.05), residuals are random, and Cook’s D < 1 for all points, so the model fits well and assumptions appear valid.”

#----------------------------------------------------------------------------

13. QUICK DEFAULT PHRASES

#----------------------------------------------------------------------------

“There is evidence of a linear relationship between x and y.”

“Predictor x is statistically significant at alpha = 0.05.”

“Residuals appear randomly scattered, indicating constant variance.”

“The model explains a large proportion of variability in y.”

“Observation 14 is potentially influential and should be checked.”

#----------------------------------------------------------------------------
