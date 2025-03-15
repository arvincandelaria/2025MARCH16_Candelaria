# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 05:51:03 2025

@author: Arvin Candelaria
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Simulated data: Lifespan of 20 randomly selected bulbs
np.random.seed(42)
true_mu = 1020  # The actual mean lifespan
sample_data = np.random.normal(true_mu, 50, size=20)

# Known standard deviation
sigma = 50
n = len(sample_data)

# Prior belief: Normal distribution (mean=1000, std=50)
prior_mu = 1000
prior_sigma = 50

# Bayesian update formulas for normal likelihood & normal prior
posterior_mu = (prior_mu / prior_sigma**2 + np.sum(sample_data) / sigma**2) / (1 / prior_sigma**2 + n / sigma**2)
posterior_sigma = np.sqrt(1 / (1 / prior_sigma**2 + n / sigma**2))

# Generate prior and posterior distributions
x = np.linspace(950, 1100, 1000)
prior_dist = stats.norm.pdf(x, prior_mu, prior_sigma)
posterior_dist = stats.norm.pdf(x, posterior_mu, posterior_sigma)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(x, prior_dist, label="Prior Distribution (Before Data)", linestyle="dashed")
plt.plot(x, posterior_dist, label="Posterior Distribution (After Data)", color="green")
plt.axvline(posterior_mu, color='red', linestyle="--", label=f"Posterior Mean: {posterior_mu:.2f}")
plt.xlabel("Lifespan of Bulbs (hours)")
plt.ylabel("Density")
plt.title("Bayesian Inference for Bulb Lifespan")
plt.legend()
plt.show()

# Print results
print(f"Posterior Mean (Updated belief of true lifespan): {posterior_mu:.2f} hours")
print(f"Posterior Standard Deviation: {posterior_sigma:.2f} hours")
