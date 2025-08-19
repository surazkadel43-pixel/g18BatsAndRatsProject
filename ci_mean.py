import scipy.stats as st
import math

# Method 1: Step-by-step calculation
# sample mean, standard deviation, and sample size
x_bar = 175
s = 20
n = 40
print("Mean: %.2f. Standard deviation: %.2f. Size: %d." % (x_bar, s, n))

# z-score (assuming 95% Confidence Level)
z_score = st.norm.ppf(q = 0.975)
print("Z-statistic: %.2f" % z_score)

# compute standard error
std_err = s / math.sqrt(n)
print("Standard error: %.2f" % std_err)

# compute the margin of error
mrg_err = z_score * std_err
print("Margin of error: %.2f" % mrg_err)

# get the lower and upper bound of confidence interval
ci_low = x_bar - mrg_err
ci_upp = x_bar + mrg_err

print("Confidence Interval of the mean: %.2f to %.2f" % (ci_low, ci_upp))


# Method 2: Use the statsmodels package
import statsmodels.stats.weightstats as stm

ci_low_stm, ci_upp_stm = stm._zconfint_generic(x_bar,std_err,alpha=0.05, alternative="two-sided")
print("Confidence Interval of the mean: %.2f to %.2f" % (ci_low_stm, ci_upp_stm))
