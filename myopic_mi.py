import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat


def symm(M):
    return 0.5 * (M + M.T)


def posterior_cov(P, A):
    # Equivalent to inv(inv(P) + A)
    return symm(np.linalg.solve(np.eye(2) + P @ A, P))


def mi_attention_from_A(P, A):
    return 0.5 * np.log(np.linalg.det(np.eye(2) + P @ A))


def action_from_L(l1, l2, l3):
    L = np.array([[l1, 0.0],
                  [l2, l3]], dtype=float)
    return symm(L @ L.T)


def admissible_l2_roots_mi(P, alpha, l1, l3, l1sq, l3sq, disc_tol):
    # For MI:
    #   0.5*log(det(I + P*A)) = alpha
    # with A = L L', L = [l1 0; l2 l3].
    #
    # This reduces to a quadratic in l2:
    #   a l2^2 + b l2 + c = 0

    p11 = P[0, 0]
    p12 = P[0, 1]
    p22 = P[1, 1]
    detP = np.linalg.det(P)

    a = p22
    b = 2.0 * p12 * l1
    c = 1.0 + p11 * l1sq + p22 * l3sq + detP * (l1sq * l3sq) - np.exp(2.0 * alpha)

    disc = b * b - 4.0 * a * c

    if disc < -disc_tol:
        return []

    disc = max(disc, 0.0)
    s = np.sqrt(disc)

    r1 = (-b - s) / (2.0 * a)
    r2 = (-b + s) / (2.0 * a)

    if abs(r1 - r2) < 1e-14:
        return [r1]
    return [r1, r2]


def lex_less(a, b, tol):
    for ai, bi in zip(a, b):
        if ai < bi - tol:
            return True
        elif ai > bi + tol:
            return False
    return False


def solve_stage_mi_exact_grid(P, alpha, loss_fn, lmin, lmax, grid_step, disc_tol, mi_tol, tie_tol):
    lvals = np.arange(lmin, lmax + 1e-12, grid_step)

    found_any = False
    A_best = np.zeros((2, 2))
    P_best = P.copy()
    val_best = np.inf

    # deterministic tie-breaker state
    best_key = [np.inf, np.inf, np.inf, np.inf, np.inf]

    for l1 in lvals:
        l1sq = l1 * l1

        for l3 in lvals:
            l3sq = l3 * l3

            roots_l2 = admissible_l2_roots_mi(P, alpha, l1, l3, l1sq, l3sq, disc_tol)

            if len(roots_l2) == 0:
                continue

            for l2 in roots_l2:
                if not math.isfinite(l2):
                    continue

                A = action_from_L(l1, l2, l3)
                mi = mi_attention_from_A(P, A)

                if abs(mi - alpha) > mi_tol:
                    continue

                Pn = posterior_cov(P, A)
                val = loss_fn(Pn)

                # deterministic tie-break:
                #   1) stage loss
                #   2) trace(Pn)
                #   3) l1
                #   4) abs(l2)
                #   5) l3
                cand_key = [val, np.trace(Pn), l1, abs(l2), l3]

                if (not found_any) or lex_less(cand_key, best_key, tie_tol):
                    found_any = True
                    A_best = A
                    P_best = Pn
                    val_best = val
                    best_key = cand_key

    if not found_any:
        raise RuntimeError("No feasible action found at this stage.")

    return A_best, P_best, val_best


if __name__ == "__main__":
    # Exact draft-matching Python code for myopic MI experiments
    # Faithful to the MATLAB script:
    #   - T = 10
    #   - P0 = Q diag(5,1) Q'
    #   - equal per-stage attention alpha = gamma/T
    #   - MI attention: G(P_t,P_{t-1}) = 0.5*log|P_{t-1} P_t^{-1}| = gamma/T
    #   - A_t = L L' with L = [l1 0; l2 l3]
    #   - (l1,l3) sampled on uniform grid [0,2] with step 0.0005
    #   - l2 obtained from the per-stage MI equality
    #   - Scenario 1: trace(P)
    #   - Scenario 2: sum_{i != j} |P_ij| = 2*abs(P(1,2))
    #   - Scenario 3: ||P||_F
    #   - Total cost and regret evaluated with trace
    #   - Instant-cost plot evaluated with each scenario's own stage distortion
    #
    # WARNING:
    #   The exact grid step 0.0005 is extremely expensive.
    #   This script is faithful to the MATLAB version, but may take a very long time.

    # ----------------------------
    # User controls
    # ----------------------------
    gamma_list = np.arange(0.5, 2.5 + 1e-12, 0.5)
    T = 10
    gamma_target = 0.5

    grid_step = 5e-4   # exact draft setting
    # grid_step = 5e-3  # use this temporarily if you want faster testing

    lmin = 0.0
    lmax = 2.0

    disc_tol = 1e-12
    mi_tol = 1e-10
    tie_tol = 1e-14

    save_results = False
    results_file = "myopic_MI_exact_draft_results.mat"

    save_figures = True
    show_figures = True

    # ----------------------------
    # Initial covariance P0
    # ----------------------------
    theta = 30.0 * np.pi / 180.0
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]], dtype=float)
    Lambda = np.diag([5.0, 1.0])
    P0 = symm(Q @ Lambda @ Q.T)

    # ----------------------------
    # Stage losses
    # ----------------------------
    L1_trace = lambda P: np.trace(P)
    L2_offdiag = lambda P: 2.0 * abs(P[0, 1])
    L3_frob = lambda P: np.linalg.norm(P, ord="fro")

    scenarios = [
        {"name": "Scenario 1 (Benchmark)", "short": "L^{(1)}", "loss_fn": L1_trace},
        {"name": "Scenario 2", "short": "L^{(2)}", "loss_fn": L2_offdiag},
        {"name": "Scenario 3", "short": "L^{(3)}", "loss_fn": L3_frob},
    ]

    S = len(scenarios)
    G = len(gamma_list)

    # ----------------------------
    # Storage
    # ----------------------------
    trace_totals = np.zeros((S, G))       # J^[m](gamma) = sum_t trace(P_t^[m])
    trace_paths = np.zeros((S, G, T))     # trace(P_t^[m])
    inst_costs = np.zeros((S, G, T))      # L^(m)(P_t^[m])
    P_paths = [[None for _ in range(G)] for _ in range(S)]   # full covariance paths
    A_paths = [[None for _ in range(G)] for _ in range(S)]   # full action paths

    # ----------------------------
    # Main loop
    # ----------------------------
    for ig, gamma in enumerate(gamma_list):
        alpha = gamma / T

        print("\n====================================================")
        print(f"gamma = {gamma:.2f}, alpha = {alpha:.6f}")
        print("====================================================")

        for s in range(S):
            print(f"\n{scenarios[s]['name']}")

            P_seq = [None] * (T + 1)
            A_seq = [None] * T
            P_seq[0] = P0.copy()

            Pcur = P0.copy()

            for t in range(T):
                print(f"  Stage t = {t+1} ...")

                Astar, Pnext, stage_val = solve_stage_mi_exact_grid(
                    Pcur, alpha, scenarios[s]["loss_fn"],
                    lmin, lmax, grid_step, disc_tol, mi_tol, tie_tol
                )

                A_seq[t] = Astar
                P_seq[t + 1] = Pnext

                trace_paths[s, ig, t] = np.trace(Pnext)
                inst_costs[s, ig, t] = stage_val

                print(f"    trace(P_t) = {np.trace(Pnext):.10f}")
                print(f"    inst cost  = {stage_val:.10f}")
                print(f"    MI         = {mi_attention_from_A(Pcur, Astar):.10f}")

                Pcur = Pnext

            trace_totals[s, ig] = np.sum(trace_paths[s, ig, :])
            P_paths[s][ig] = P_seq
            A_paths[s][ig] = A_seq

            print(f"  Total trace cost = {trace_totals[s, ig]:.10f}")

    # ----------------------------
    # Regret: trace-based vs benchmark
    # ----------------------------
    regrets = np.zeros((S, G, T))
    for ig in range(G):
        bench = trace_paths[0, ig, :]
        for s in range(1, S):
            regrets[s, ig, :] = (trace_paths[s, ig, :] - bench) / bench

    # ----------------------------
    # Plot index for gamma_target
    # ----------------------------
    idx_gamma = int(np.argmin(np.abs(gamma_list - gamma_target)))
    tvec = np.arange(1, T + 1)

    # ----------------------------
    # Figure 1: Total cost vs gamma
    # ----------------------------
    plt.figure(facecolor="white")
    plt.plot(gamma_list, trace_totals[0, :], "-o", linewidth=2.5,
             label=r"$\mathcal{L}^{(1)}$ (Benchmark)")
    plt.plot(gamma_list, trace_totals[1, :], "--o", linewidth=2.5,
             label=r"$\mathcal{L}^{(2)}$")
    plt.plot(gamma_list, trace_totals[2, :], "-.o", linewidth=2.5,
             label=r"$\mathcal{L}^{(3)}$")
    plt.xlabel(r"$\gamma$")
    plt.ylabel("Total cost")
    plt.title(r"Total cost vs $\gamma$ using MI attention")
    plt.legend(loc="best")
    plt.grid(True)
    plt.box(True)
    plt.gca().tick_params(labelsize=14)
    plt.tight_layout()
    if save_figures:
        plt.savefig("figure1_total_cost_vs_gamma.png", dpi=300, bbox_inches="tight")

    # ----------------------------
    # Figure 2: Regret vs t at gamma = 2.5
    # ----------------------------
    plt.figure(facecolor="white")
    plt.plot(tvec, 100.0 * regrets[1, idx_gamma, :], "--o", linewidth=2.5,
             label=r"$\mathcal{L}^{(2)}$")
    plt.plot(tvec, 100.0 * regrets[2, idx_gamma, :], "-.o", linewidth=2.5,
             label=r"$\mathcal{L}^{(3)}$")
    plt.xlabel(r"$t$")
    plt.ylabel("Regret (%)")
    plt.title(r"Regret vs $t$ at $\gamma=2.5$ using MI attention")
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlim([1, T])
    plt.box(True)
    plt.gca().tick_params(labelsize=14)
    plt.tight_layout()
    if save_figures:
        plt.savefig("figure2_regret_vs_t_gamma_2_5.png", dpi=300, bbox_inches="tight")

    # ----------------------------
    # Figure 3: Instant stage costs vs t at gamma = 2.5
    # ----------------------------
    plt.figure(facecolor="white")
    plt.plot(tvec, inst_costs[0, idx_gamma, :], "-o", linewidth=2.5,
             label=r"$\mathcal{L}^{(1)}$ (Benchmark)")
    plt.plot(tvec, inst_costs[1, idx_gamma, :], "--o", linewidth=2.5,
             label=r"$\mathcal{L}^{(2)}$")
    plt.plot(tvec, inst_costs[2, idx_gamma, :], "-.o", linewidth=2.5,
             label=r"$\mathcal{L}^{(3)}$")
    plt.xlabel(r"$t$")
    plt.ylabel("Instant cost")
    plt.title(r"Instant stage costs vs $t$ at $\gamma=2.5$ using MI attention")
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlim([1, T])
    plt.box(True)
    plt.gca().tick_params(labelsize=14)
    plt.tight_layout()
    if save_figures:
        plt.savefig("figure3_instant_cost_vs_t_gamma_2_5.png", dpi=300, bbox_inches="tight")

    # ----------------------------
    # Save .mat results if requested
    # ----------------------------
    results = {
        "gamma_list": gamma_list,
        "T": T,
        "P0": P0,
        "trace_totals": trace_totals,
        "trace_paths": trace_paths,
        "inst_costs": inst_costs,
        "regrets": regrets,
        "grid_step": grid_step,
        "gamma_target": gamma_target,
    }

    if save_results:
        savemat(results_file, results)

    # ----------------------------
    # IMPORTANT: actually display the figures
    # ----------------------------
    if show_figures:
        plt.show()
