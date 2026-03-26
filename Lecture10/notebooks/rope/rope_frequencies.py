import numpy as np
import matplotlib.pyplot as plt

d = 8  # small embedding for clarity
positions = np.arange(0, 64)

# theta for each pair
i = np.arange(d // 2)           # [0, 1, 2, 3]
theta = 10000 ** (-2 * i / d)   # decreasing frequencies

print("Frequencies per pair:")
for idx, t in enumerate(theta):
    print(f"  pair {idx}: theta = {t:.4f}  (full cycle every {2*np.pi/t:.1f} tokens)")

# angle phi[m, i] = m * theta[i]
phi = np.outer(positions, theta)  # (64, 4)

# what each pair "sees": cos of relative distance (m - n) * theta_i
# fix n=0, vary m — this is what dot product sees for pair i
cos_vals = np.cos(phi)  # (64, 4)

fig, axes = plt.subplots(2, 1, figsize=(12, 7))

# --- top: cos signal per pair ---
ax = axes[0]
for idx in range(d // 2):
    ax.plot(positions, cos_vals[:, idx],
            label=f"pair {idx}  θ={theta[idx]:.3f}", linewidth=1.8)
ax.set_title("cos(m · θᵢ) per dimension pair  —  what the dot product sees as m varies (n=0 fixed)")
ax.set_xlabel("position m")
ax.set_ylabel("cos(m · θᵢ)")
ax.legend(loc="upper right")
ax.axhline(0, color="gray", linewidth=0.5)
ax.grid(True, alpha=0.3)

# --- bottom: 2D rotation of a fixed unit vector per pair ---
ax2 = axes[1]
unit = np.array([1.0, 0.0])
sample_positions = [0, 1, 2, 4, 8, 16, 32]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(sample_positions)))

for pair_idx in range(d // 2):
    ax2.set_visible(False)  # will use subplots below

fig2, axes2 = plt.subplots(1, d // 2, figsize=(14, 3.5))
fig2.suptitle("2D rotation of unit vector per pair, for positions 0,1,2,4,8,16,32", fontsize=12)

for pair_idx in range(d // 2):
    ax = axes2[pair_idx]
    for pos, col in zip(sample_positions, colors):
        angle = pos * theta[pair_idx]
        rotated = np.array([np.cos(angle), np.sin(angle)])
        ax.annotate("", xy=rotated, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.8))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_title(f"pair {pair_idx}\nθ={theta[pair_idx]:.3f}", fontsize=9)
    ax.axhline(0, color="gray", lw=0.4)
    ax.axvline(0, color="gray", lw=0.4)
    ax.grid(True, alpha=0.2)

# colorbar legend
sm = plt.cm.ScalarMappable(cmap="plasma",
                            norm=plt.Normalize(vmin=0, vmax=sample_positions[-1]))
sm.set_array([])
fig2.colorbar(sm, ax=axes2, orientation="vertical", label="position m", shrink=0.8)

fig.tight_layout()
fig2.tight_layout()
fig.savefig("/mnt/user-data/outputs/rope_cos_signals.png", dpi=130)
fig2.savefig("/mnt/user-data/outputs/rope_rotations.png", dpi=130)
print("\nSaved: rope_cos_signals.png, rope_rotations.png")
