import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# load posterior draws
df = pd.read_csv("posterior_draws_samples.csv")

tab20 = cm.get_cmap('tab20')

params = [
    ("D_scale", r"$\theta_0$", tab20(2)),   # red-ish
    ("cap", r"$\theta_1$", tab20(2)),   # blue-ish
    ("lambda_scale", r"$\theta_2$", tab20(2)),   # green-ish
]
siglabel=r"$\sigma$"
for col, label, c in params:
    plt.figure()
    plt.hist(df[col], bins=60, color=c, edgecolor="black", alpha=0.6)
    plt.title(f'Posterior samples: {label} with {siglabel} = 0.001 ')
    plt.xlabel(label)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

