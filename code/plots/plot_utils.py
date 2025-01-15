import numpy as np
def confidence_interval(std_dev, n, confidence_level=0.95):
    import scipy.stats as stats

    z_value = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    se = std_dev / np.sqrt(n)

    return z_value * se
