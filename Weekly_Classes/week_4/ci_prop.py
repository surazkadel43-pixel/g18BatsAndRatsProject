import statsmodels.stats.proportion as stm 

# the number of objects of interest in a sample
prop = 52

# sample size
total = 100

# confidence level
conf_lvl = 0.95

# significance level (alpha)
sig_lvl = 1 - conf_lvl

# compute confidence interval for proportion (confidence level=95%)
ci_low, ci_upp = stm.proportion_confint(prop, total, alpha=sig_lvl, method="normal")
print("C.I. of proportion at %d%% confidence level is %.3f (%.1f%%) and %.3f (%.1f%%)." % 
      (conf_lvl*100, ci_low, ci_low*100, ci_upp, ci_upp*100))