# Re-run in fresh kernel: implement full 20‑day MPC via LSMC backward induction

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
H = 20        # horizon in days
M = 2500      # Monte Carlo paths
sigma = 0.02  # volatility proxy for TE
lam = 0.001   # cost weight
u_max = 0.05
actions = np.linspace(-u_max, u_max, 11)   # discrete trade grid
kappa = 100.0   # terminal penalty weight

# Random index drift (eps) each day for each path
eps = np.random.normal(scale=0.02, size=(M, H))

# === Forward simulation of baseline states (no trades, just eps) =========
d_baseline = np.zeros((M, H+1))
for t in range(H):
    d_baseline[:, t+1] = d_baseline[:, t] - eps[:, t]

# Terminal cost baseline
V_next = kappa * sigma**2 * d_baseline[:, -1]**2

coeffs = []
policy = np.zeros((M, H))   # optimal action per path & time

for t in reversed(range(H)):
    d_t = d_baseline[:, t]

    # Fit quadratic proxy for value function at time t
    X = np.column_stack([np.ones(M), d_t, d_t**2])
    coeff = np.linalg.lstsq(X, V_next, rcond=None)[0]
    coeffs.insert(0, coeff)
    a, b, c = coeff

    best_val = np.full(M, np.inf)
    best_u = np.zeros(M)

    for u in actions:
        d_after = d_t + u
        stage_te = sigma**2 * d_after**2
        stage_cost = lam * np.abs(u)
        # approximate expected future cost via regression
        V_hat = a + b*d_after + c*d_after**2
        tot = stage_te + stage_cost + V_hat
        mask = tot < best_val
        best_val[mask] = tot[mask]
        best_u[mask] = u

    policy[:, t] = best_u
    # simulate realised next state under chosen u and eps
    d_next = d_t + best_u - eps[:, t]
    V_next = best_val  # already includes future cost proxy
    d_baseline[:, t+1] = d_next  # update baseline path to follow policy

# === Evaluate policy with fresh simulation ================================
eps_eval = np.random.normal(scale=0.02, size=(M, H))
d_sim = np.zeros((M, H+1))
te_accum = np.zeros(M)
cost_accum = np.zeros(M)

for t in range(H):
    # determine control via nearest-neighbour on coeffs (closed-form)
    a, b, c = coeffs[t]
    d = d_sim[:, t]
    # Evaluate each candidate action and pick best (like during training)
    best_u = np.zeros(M)
    best_val = np.full(M, np.inf)
    for u in actions:
        d_after = d + u
        val = sigma**2*d_after**2 + lam*np.abs(u) + (a + b*d_after + c*d_after**2)
        mask = val < best_val
        best_val[mask] = val[mask]
        best_u[mask] = u
    # apply best_u
    cost_accum += lam*np.abs(best_u)
    d_after = d + best_u
    te_accum += sigma**2 * d_after**2
    d_sim[:, t+1] = d_after - eps_eval[:, t]

print("20‑day horizon MPC results (Monte‑Carlo eval,", M, "paths)")
print("Average cumulative TE² :", te_accum.mean())
print("Average cumulative trading cost :", cost_accum.mean())

plt.plot(d_sim[0], marker='o')
plt.title("Sample active‑weight path under trained MPC policy")
plt.xlabel("Day")
plt.ylabel("Active weight d_t")
plt.grid()
plt.show()
