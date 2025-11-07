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
