import scipy.stats as st
import numpy as np

# sample values
sample = np.array([248,37, 146, 19, 66, 236, 164, 30, 13, 144, 242, 20])

# compute mean and standard deviation of the sample
print("Computing the basic statistics ...")
x_bar = st.tmean(sample)
s = st.tstd(sample)
print("\t Sample mean: %.2f" % x_bar)
print("\t Sample std. dev.: %.2f" % s)

# perform one-sample t-test
# null hypothesis: population mean = 88
# alternative hypothesis: population mean > 88 (in the function below, note the argument 'greater')
t_stats, p_val = st.ttest_1samp(sample, 88, alternative='greater')
print("\n Computing t* ...")
print("\t t-statistic (t*): %.2f" % t_stats)

print("\n Computing p-value ...")
print("\t p-value: %.4f" % p_val)

print("\n Conclusion:")
if p_val < 0.05:
    print("\t We reject the null hypothesis.")
else:
    print("\t We accept the null hypothesis.")