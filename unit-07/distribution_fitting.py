import numpy as np
import matplotlib.pyplot as plt
from fitter import Fitter
import os

np.random.seed(42)
n_samples = 50
save_path = "./unit-07/output"
os.makedirs(save_path, exist_ok=True)
test_dists = ["norm", "gamma", "lognorm", "beta", "expon"]

# Generate Distributions
normal_data = np.random.normal(loc=0, scale=1, size=n_samples)
exp_data = np.random.exponential(scale=1, size=n_samples)
lognorm_data = np.random.lognormal(mean=0, sigma=0.5, size=n_samples)

# Fit distributions
f_normal = Fitter(normal_data, distributions=test_dists)
f_normal.fit()
print("\n\nFitting Normal Distribution Sample")
print("-"*40)
print(f_normal.summary())
print(f"\nBest fitting distribution: {f_normal.get_best(method='sumsquare_error')}")

f_exp = Fitter(exp_data, distributions=test_dists)
f_exp.fit()
print("\n\nFitting Exponential Distribution Sample")
print("-"*40)
print(f_exp.summary())
print(f"\nBest fitting distribution: {f_exp.get_best(method='sumsquare_error')}")

f_lognorm = Fitter(lognorm_data, distributions=test_dists)
f_lognorm.fit()
print("\n\nFitting Log-Normal Distribution Sample")
print("-"*40)
print(f_lognorm.summary())
print(f"\nBest fitting distribution: {f_lognorm.get_best(method='sumsquare_error')}")

# Plotting all in one figure
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Normal
plt.sca(axs[0])  # activa el subplot 0
f_normal.plot_pdf()
axs[0].hist(normal_data, bins=30, edgecolor="black", alpha=0.7, density=True)
axs[0].set_title("Normal Distribution Sample")
axs[0].set_xlabel("Value")
axs[0].set_ylabel("Density")

# Exponential
plt.sca(axs[1])
f_exp.plot_pdf()
axs[1].hist(exp_data, bins=30, edgecolor="black", alpha=0.7, density=True)
axs[1].set_title("Exponential Distribution Sample")
axs[1].set_xlabel("Value")
axs[1].set_ylabel("Density")

# Log-Normal
plt.sca(axs[2])
f_lognorm.plot_pdf()
axs[2].hist(lognorm_data, bins=30, edgecolor="black", alpha=0.7, density=True)
axs[2].set_title("Log-Normal Distribution Sample")
axs[2].set_xlabel("Value")
axs[2].set_ylabel("Density")

plt.tight_layout()
plt.savefig(f"{save_path}/distributions_fit_{n_samples}.jpg", bbox_inches="tight")
plt.show()
