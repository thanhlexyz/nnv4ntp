import numpy as np
import matplotlib.pyplot as plt

alpha = 1.0
epsilon = 0.1
xa = epsilon * alpha  # 0.01

x = np.linspace(-1, 1, 800)
y = np.empty_like(x)

mask1 = (x <= 0)
mask2 = (x > 0) & (x <= xa)
mask3 = (x > xa)

y[mask1] = alpha - epsilon * x[mask1]
y[mask2] = alpha - (1.0/epsilon) * x[mask2]
y[mask3] = x[mask3] - alpha * epsilon

plt.figure(figsize=(7, 4.5), dpi=160)
plt.plot(x, y, color='royalblue', lw=2, label=r"$\ell'(x)$ for $\alpha=0.1,\, \epsilon=0.1$")
plt.axvline(0, color='gray', lw=1, ls='--')
plt.axvline(xa, color='gray', lw=1, ls='--')

# Boundary values (for quick checks)
l0_left  = alpha - epsilon * 0.0
l0_right = alpha - (1.0/epsilon) * 0.0
lxa_left = alpha - (1.0/epsilon) * xa
lxa_right= xa - alpha * epsilon

plt.scatter([0, 0, xa, xa], [l0_left, l0_right, lxa_left, lxa_right], color=['crimson','forestgreen','crimson','forestgreen'], zorder=3)
plt.title(f"Piecewise derivative $\\ell'(x)$ with $\\{alpha=}$ and $\\{epsilon=}$")
plt.xlabel("x")
plt.ylabel(r"$\ell'(x)$")
plt.grid(True, ls=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('deepcog_loss.jpg')
