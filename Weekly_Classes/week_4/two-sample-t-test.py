import scipy.stats as st

# we are given a set basic statistics of two samples to be tested, instead of raw sample values.
# the basic statistics of sample 1:
x_bar1 = 105.32
s1 = 14.68
n1 = 57

# the basic statistics of sample 2:
x_bar2 = 96.82
s2 = 14.26
n2 = 17

# perform two-sample t-test
# null hypothesis: mean of sample 1 = mean of sample 2
# alternative hypothesis: mean of sample 1 does not equal mean of sample 2 (two-sided test)
# note the argument equal_var=False, which assumes that two populations do not have equal variance
t_stats, p_val = st.ttest_ind_from_stats(x_bar1, s1, n1, x_bar2, s2, n2, equal_var=False, alternative='two-sided')
print("\n Computing t* ...")
print("\t t-statistic (t*): %.2f" % t_stats)

print("\n Computing p-value ...")
print("\t p-value: %.4f" % p_val)

print("\n Conclusion:")
if p_val < 0.05:
    print("\t We reject the null hypothesis.")
else:
    print("\t We accept the null hypothesis.")