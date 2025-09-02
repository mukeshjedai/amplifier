import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Use the raw GitHub link, not the repo page link
amp_path = "https://raw.githubusercontent.com/mukeshjedai/amplifier/main/amplifier.txt"

# Load file directly from GitHub
amp = pd.read_csv(amp_path, delim_whitespace=True, header=None,
                  names=["freq_col", "gain", "dgain"])
amp = amp.apply(pd.to_numeric, errors="coerce").dropna()

# Convert frequency column to Hz
f_raw = amp["freq_col"].to_numpy(dtype=float)
G = amp["gain"].to_numpy(dtype=float)

# Scale to Hz if needed
if np.nanmax(f_raw) < 1e3:
    f = f_raw * 1e4
else:
    f = f_raw

# Sort by frequency
idx = np.argsort(f)
f = f[idx]
G = G[idx]

# Cumulative trapezoidal integration of G^2
cum_int = np.zeros_like(f, dtype=float)
for i in range(1, len(f)):
    df = f[i] - f[i-1]
    cum_int[i] = cum_int[i-1] + 0.5 * (G[i]**2 + G[i-1]**2) * df

# Plot
plt.figure()
plt.plot(f, cum_int, label=r"$\int_0^{f_c} G^2(f)\,df$")
plt.xlabel("Cutoff frequency $f_c$ (Hz)")
plt.ylabel(r"Cumulative $(V_{\rm rms})^2$ (arb. units)")
plt.title(r"Cumulative $(V_{\rm rms})^2$ vs cutoff frequency")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.tight_layout()
plt.show()
