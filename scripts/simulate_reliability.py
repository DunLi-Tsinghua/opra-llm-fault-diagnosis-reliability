from __future__ import annotations

import csv
import math
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "simulation_config.yaml"
FIG_DIR = PROJECT_ROOT / "figures"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
for directory in (FIG_DIR, DATA_DIR, RESULTS_DIR):
    directory.mkdir(exist_ok=True)


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

SEED = int(CONFIG["seed"])
RNG = np.random.default_rng(SEED)

FIG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DOMAIN_NAMES = [domain["name"] for domain in CONFIG["domains"]]
DOMAINS = [domain["label"] for domain in CONFIG["domains"]]
FAILURE_MODES = CONFIG["failure_modes"]

D = len(DOMAINS)
M = len(FAILURE_MODES)
N_PER_DOMAIN = np.array(CONFIG["validation_sample_sizes"], dtype=int)
BASE_PI = np.array(CONFIG["profiles"]["nominal"], dtype=float)
STRESS_PI = np.array(CONFIG["profiles"]["stressed"], dtype=float)
TRUE_THETA = np.array(CONFIG["true_theta"], dtype=float)

FAILURE_PRIOR_SHARE = np.array(CONFIG["prior"]["failure_prior_share"], dtype=float)
CLASSICAL_PRIOR_SUCCESS = np.array(CONFIG["prior"]["classical_success"], dtype=float)
CLASSICAL_PRIOR_ETA = np.array(CONFIG["prior"]["classical_eta"], dtype=float)
IMPRECISE_Q0_LOW = np.array(CONFIG["prior"]["imprecise_q0_low"], dtype=float)
IMPRECISE_Q0_HIGH = np.array(CONFIG["prior"]["imprecise_q0_high"], dtype=float)
IMPRECISE_ETA_LOW = np.array(CONFIG["prior"]["imprecise_eta_low"], dtype=float)
IMPRECISE_ETA_HIGH = np.array(CONFIG["prior"]["imprecise_eta_high"], dtype=float)
SEVERITY_WEIGHTS = np.array(CONFIG["severity_weights"], dtype=float)

R_REQ = float(CONFIG["requirements"]["R_req"])
TAU_SAFE = float(CONFIG["requirements"]["tau_safe"])
MISSION_N_SHIFT = int(CONFIG["mission_lengths"]["profile_shift"])
MISSION_N_PLANNING = int(CONFIG["mission_lengths"]["validation_planning"])
UNSAFE_MISSION_LENGTHS = [int(n) for n in CONFIG["mission_lengths"]["unsafe_screening"]]
PLANNING_GRID = CONFIG["validation_planning_grid"]
_COMB_CACHE: dict[int, np.ndarray] = {}


def comb_matrix(n_max: int) -> np.ndarray:
    if n_max not in _COMB_CACHE:
        mat = np.zeros((n_max + 1, n_max + 1), dtype=float)
        mat[0, 0] = 1.0
        for n in range(1, n_max + 1):
            mat[n, 0] = 1.0
            mat[n, n] = 1.0
            mat[n, 1:n] = mat[n - 1, : n - 1] + mat[n - 1, 1:n]
        _COMB_CACHE[n_max] = mat
    return _COMB_CACHE[n_max]


def build_alpha(q0: np.ndarray, strength: np.ndarray) -> np.ndarray:
    alpha = np.zeros((D, M + 1), dtype=float)
    alpha[:, 0] = strength * q0
    alpha[:, 1:] = strength[:, None] * (1.0 - q0[:, None]) * FAILURE_PRIOR_SHARE[None, :]
    return alpha


def simulate_counts() -> np.ndarray:
    return np.vstack([RNG.multinomial(int(N_PER_DOMAIN[d]), TRUE_THETA[d]) for d in range(D)]).astype(float)


def posterior_reliability_curve(
    counts: np.ndarray,
    alpha: np.ndarray,
    pi: np.ndarray,
    n_values: np.ndarray,
) -> np.ndarray:
    n_max = int(np.max(n_values))
    comb = comb_matrix(n_max)
    post = counts + alpha
    post_total = post.sum(axis=1)
    moments = np.zeros((D, n_max + 1), dtype=float)
    moments[:, 0] = 1.0
    for d in range(D):
        for k in range(1, n_max + 1):
            moments[d, k] = moments[d, k - 1] * (post[d, 0] + k - 1) / (post_total[d] + k - 1)

    total_moments = np.zeros(n_max + 1, dtype=float)
    total_moments[0] = 1.0
    for d in range(D):
        component = np.array([(pi[d] ** k) * moments[d, k] for k in range(n_max + 1)])
        updated = np.zeros(n_max + 1, dtype=float)
        for n in range(n_max + 1):
            idx = np.arange(n + 1)
            updated[n] = np.dot(comb[n, : n + 1], total_moments[n - idx] * component[: n + 1])
        total_moments = updated

    return np.array([total_moments[int(n)] for n in n_values])


def point_reliability_curve(per_domain_success: np.ndarray, pi: np.ndarray, n_values: np.ndarray) -> np.ndarray:
    single_demand = float(pi @ per_domain_success)
    return single_demand ** n_values


def alpha_vertices(
    q0_low: np.ndarray = IMPRECISE_Q0_LOW,
    q0_high: np.ndarray = IMPRECISE_Q0_HIGH,
    eta_low: np.ndarray = IMPRECISE_ETA_LOW,
    eta_high: np.ndarray = IMPRECISE_ETA_HIGH,
):
    for q_bits in product([0, 1], repeat=D):
        q0 = np.where(np.array(q_bits, dtype=bool), q0_high, q0_low)
        for use_high_strength in [False, True]:
            strength = eta_high if use_high_strength else eta_low
            yield build_alpha(q0, strength)


def imprecise_envelope(
    counts: np.ndarray,
    pi: np.ndarray,
    n_values: np.ndarray,
    q0_low: np.ndarray = IMPRECISE_Q0_LOW,
    q0_high: np.ndarray = IMPRECISE_Q0_HIGH,
    eta_low: np.ndarray = IMPRECISE_ETA_LOW,
    eta_high: np.ndarray = IMPRECISE_ETA_HIGH,
) -> tuple[np.ndarray, np.ndarray]:
    curves = [
        posterior_reliability_curve(counts, alpha, pi, n_values)
        for alpha in alpha_vertices(q0_low, q0_high, eta_low, eta_high)
    ]
    stacked = np.vstack(curves)
    return stacked.min(axis=0), stacked.max(axis=0)


def profile_envelope(counts: np.ndarray, profiles: list[np.ndarray], n_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    curves = []
    for pi in profiles:
        for alpha in alpha_vertices():
            curves.append(posterior_reliability_curve(counts, alpha, pi, n_values))
    stacked = np.vstack(curves)
    return stacked.min(axis=0), stacked.max(axis=0)


def severity_weighted_plugin_curve(counts: np.ndarray, alpha: np.ndarray, pi: np.ndarray, n_values: np.ndarray) -> np.ndarray:
    post_mean = (counts + alpha) / (counts + alpha).sum(axis=1, keepdims=True)
    domain_score = 1.0 - post_mean[:, 1:] @ SEVERITY_WEIGHTS
    return point_reliability_curve(domain_score, pi, n_values)


def linear_dirichlet_moments(alpha: np.ndarray, coefficients: np.ndarray, n_max: int) -> np.ndarray:
    """Moments of (coefficients dot theta)^k for theta ~ Dirichlet(alpha)."""
    coeff_poly = np.zeros(n_max + 1, dtype=float)
    coeff_poly[0] = 1.0
    for a_i, c_i in zip(alpha, coefficients):
        component = np.zeros(n_max + 1, dtype=float)
        component[0] = 1.0
        for k in range(1, n_max + 1):
            component[k] = component[k - 1] * (a_i + k - 1.0) * c_i / k
        coeff_poly = np.convolve(coeff_poly, component)[: n_max + 1]

    moments = np.ones(n_max + 1, dtype=float)
    scale = 1.0
    total_alpha = float(np.sum(alpha))
    for k in range(1, n_max + 1):
        scale *= k / (total_alpha + k - 1.0)
        moments[k] = coeff_poly[k] * scale
    return moments


def aggregate_predictive_from_domain_moments(
    domain_moments: np.ndarray,
    pi: np.ndarray,
    n_values: np.ndarray,
) -> np.ndarray:
    n_max = int(np.max(n_values))
    comb = comb_matrix(n_max)
    total_moments = np.zeros(n_max + 1, dtype=float)
    total_moments[0] = 1.0
    for d in range(D):
        component = np.array([(pi[d] ** k) * domain_moments[d, k] for k in range(n_max + 1)])
        updated = np.zeros(n_max + 1, dtype=float)
        for n in range(n_max + 1):
            idx = np.arange(n + 1)
            updated[n] = np.dot(comb[n, : n + 1], total_moments[n - idx] * component[: n + 1])
        total_moments = updated
    return np.array([total_moments[int(n)] for n in n_values])


def severity_weighted_predictive_curve(
    counts: np.ndarray,
    alpha: np.ndarray,
    pi: np.ndarray,
    n_values: np.ndarray,
) -> np.ndarray:
    n_max = int(np.max(n_values))
    post = counts + alpha
    # rho_d^(w)(theta_d) = theta_0 + sum_m (1 - w_m) theta_m.
    coefficients = np.concatenate(([1.0], 1.0 - SEVERITY_WEIGHTS))
    domain_moments = np.vstack(
        [linear_dirichlet_moments(post[d], coefficients, n_max) for d in range(D)]
    )
    return aggregate_predictive_from_domain_moments(domain_moments, pi, n_values)


def posterior_success_mean_bounds(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.ones(D)
    upper = np.zeros(D)
    for alpha in alpha_vertices():
        post = counts + alpha
        mean_success = post[:, 0] / post.sum(axis=1)
        lower = np.minimum(lower, mean_success)
        upper = np.maximum(upper, mean_success)
    return lower, upper


def posterior_unsafe_upper_by_domain(counts: np.ndarray) -> np.ndarray:
    upper = np.zeros(D)
    for alpha in alpha_vertices():
        post = counts + alpha
        unsafe = post[:, 4] / post.sum(axis=1)
        upper = np.maximum(upper, unsafe)
    return upper


def unsafe_upper_probability(counts: np.ndarray, profiles: list[np.ndarray]) -> float:
    unsafe_domain = posterior_unsafe_upper_by_domain(counts)
    return max(float(pi @ unsafe_domain) for pi in profiles)


def save_figure(fig: plt.Figure, name: str) -> None:
    for ext in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def first_reaches(x_values: np.ndarray, y_values: np.ndarray, threshold: float) -> float | None:
    hits = np.where(y_values >= threshold)[0]
    if len(hits) == 0:
        return None
    return float(x_values[hits[0]])


def first_falls_below(x_values: np.ndarray, y_values: np.ndarray, threshold: float) -> float | None:
    hits = np.where(y_values < threshold)[0]
    if len(hits) == 0:
        return None
    return float(x_values[hits[0]])


def draw_box(ax, xy, width, height, text, fc="#F6F6F6", ec="#333333", fontsize=9):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return box


def draw_arrow(ax, start, end, color="#333333", connectionstyle="arc3,rad=0.0"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.0,
        color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def fig1_system_concept() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    draw_box(ax, (0.04, 0.62), 0.20, 0.16, "Operational profile\n$\\Pi=(\\pi_1,\\ldots,\\pi_D)$", "#E8F1FA")
    draw_box(ax, (0.04, 0.30), 0.20, 0.16, "Future diagnosis\nrequests", "#E8F1FA")
    draw_arrow(ax, (0.14, 0.62), (0.14, 0.46))
    draw_arrow(ax, (0.24, 0.38), (0.34, 0.50))

    draw_box(ax, (0.35, 0.24), 0.30, 0.52, "", "#F7F3E8", fontsize=10)
    ax.text(0.50, 0.72, "LLM-assisted fault\n diagnosis system", ha="center", va="center", fontsize=10)
    sub_y = [0.60, 0.49, 0.38, 0.27]
    sub_text = ["Evidence retrieval", "Root-cause reasoning", "Grounding/citation", "Maintenance recommendation"]
    for y, label in zip(sub_y, sub_text):
        draw_box(ax, (0.39, y), 0.22, 0.07, label, "#FFFFFF", "#777777", fontsize=8)

    draw_arrow(ax, (0.65, 0.50), (0.76, 0.50))
    draw_box(ax, (0.77, 0.67), 0.19, 0.10, "Failure-free\n$\\theta_{d,0}$", "#E8F6EA")
    draw_box(ax, (0.77, 0.50), 0.19, 0.12, "Failure modes\n$\\theta_{d,1:M}$", "#FBEAEA")
    draw_box(ax, (0.77, 0.27), 0.19, 0.13, "Reliability envelope\n$R_{lower}(n), R_{upper}(n)$", "#EFEAF8")
    draw_arrow(ax, (0.86, 0.67), (0.86, 0.62))
    draw_arrow(ax, (0.86, 0.50), (0.86, 0.40))
    draw_arrow(ax, (0.66, 0.24), (0.77, 0.33), connectionstyle="arc3,rad=-0.08")

    ax.text(0.50, 0.12, "Assessment target: probability of no diagnostic failure over the next $n$ operational demands", ha="center", fontsize=10)
    save_figure(fig, "fig1_system_concept")


def fig2_failure_taxonomy() -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    draw_box(ax, (0.36, 0.78), 0.28, 0.12, "Diagnostic failure\n$Z_i\\neq 0$", "#EFEAF8", fontsize=11)

    positions = [(0.05, 0.43), (0.285, 0.43), (0.52, 0.43), (0.755, 0.43)]
    colors = ["#E8F1FA", "#F7F3E8", "#E8F6EA", "#FBEAEA"]
    titles = [
        "Evidence retrieval\nfailure",
        "Root-cause reasoning\nfailure",
        "Grounding/citation\nfailure",
        "Unsafe maintenance\nrecommendation",
    ]
    descriptions = [
        "Missing or irrelevant\nmanuals, logs, cases",
        "Incorrect causal chain\nor physical inconsistency",
        "Unsupported statement\nor invalid citation",
        "Action may increase\nsafety or asset risk",
    ]
    for pos, color, title, desc in zip(positions, colors, titles, descriptions):
        draw_box(ax, pos, 0.19, 0.17, title, color, fontsize=9)
        ax.text(pos[0] + 0.095, pos[1] - 0.08, desc, ha="center", va="top", fontsize=8)
        draw_arrow(ax, (0.50, 0.78), (pos[0] + 0.095, pos[1] + 0.17), connectionstyle="arc3,rad=0.05")
    ax.text(0.50, 0.17, "Failure-mode-aware counts: $y_{d,0}, y_{d,1}, y_{d,2}, y_{d,3}, y_{d,4}$", ha="center", fontsize=10)
    save_figure(fig, "fig2_failure_taxonomy")


def add_expected_validation_counts(counts: np.ndarray, total_additional: int, allocation: np.ndarray) -> np.ndarray:
    allocation = allocation / allocation.sum()
    additional = total_additional * allocation[:, None] * TRUE_THETA
    return counts + additional


def risk_targeted_allocation(counts: np.ndarray) -> np.ndarray:
    lower_success, upper_success = posterior_success_mean_bounds(counts)
    unsafe_upper = posterior_unsafe_upper_by_domain(counts)
    scores = BASE_PI * ((1.0 - lower_success) + 0.6 * (upper_success - lower_success) + 2.0 * unsafe_upper)
    if np.all(scores <= 0):
        return BASE_PI
    return scores / scores.sum()


def make_numeric_figures(counts: np.ndarray) -> dict[str, float]:
    n_values = np.arange(1, 201)
    observed_success = counts[:, 0] / counts.sum(axis=1)
    alpha_classical = build_alpha(CLASSICAL_PRIOR_SUCCESS, CLASSICAL_PRIOR_ETA)

    accuracy_curve = point_reliability_curve(observed_success, BASE_PI, n_values)
    classical_curve = posterior_reliability_curve(counts, alpha_classical, BASE_PI, n_values)
    lower_curve, upper_curve = imprecise_envelope(counts, BASE_PI, n_values)
    severity_plugin_curve = severity_weighted_plugin_curve(counts, alpha_classical, BASE_PI, n_values)
    severity_predictive_curve = severity_weighted_predictive_curve(counts, alpha_classical, BASE_PI, n_values)

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.fill_between(n_values, lower_curve, upper_curve, color="#A8C7E6", alpha=0.45, label="Imprecise Bayesian envelope")
    ax.plot(n_values, accuracy_curve, color="#4D4D4D", linestyle="--", linewidth=1.8, label="Accuracy-only point estimate")
    ax.plot(n_values, classical_curve, color="#1F77B4", linewidth=2.0, label="Classical Bayesian")
    ax.axhline(R_REQ, color="#8B1A1A", linewidth=1.2, linestyle=":", label="$R_{req}$")
    ax.set_xlabel("Future diagnosis tasks, $n$")
    ax.set_ylabel("Reliability")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "fig3_reliability_envelope")

    mission_n = np.array([MISSION_N_SHIFT])
    lambdas = np.linspace(0, 1, 31)
    acc_shift, classical_shift, lower_shift, upper_shift = [], [], [], []
    for lam in lambdas:
        pi = (1.0 - lam) * BASE_PI + lam * STRESS_PI
        acc_shift.append(point_reliability_curve(observed_success, pi, mission_n)[0])
        classical_shift.append(posterior_reliability_curve(counts, alpha_classical, pi, mission_n)[0])
        low, up = imprecise_envelope(counts, pi, mission_n)
        lower_shift.append(low[0])
        upper_shift.append(up[0])

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.fill_between(lambdas, lower_shift, upper_shift, color="#D9B38C", alpha=0.45, label="Imprecise envelope")
    ax.plot(lambdas, acc_shift, color="#4D4D4D", linestyle="--", linewidth=1.8, label="Accuracy-only")
    ax.plot(lambdas, classical_shift, color="#B35C1E", linewidth=2.0, label="Classical Bayesian")
    ax.axhline(R_REQ, color="#8B1A1A", linewidth=1.2, linestyle=":", label="$R_{req}$")
    lambda_boundary = first_falls_below(lambdas, np.array(lower_shift), R_REQ)
    if lambda_boundary is not None:
        ax.axvline(lambda_boundary, color="#8B1A1A", linewidth=1.0, linestyle="--")
        ax.text(lambda_boundary + 0.02, R_REQ + 0.012, "$\\lambda_{review}$", color="#8B1A1A", fontsize=9)
    ax.set_xlabel("Profile shift parameter, $\\lambda$")
    ax.set_ylabel(f"Reliability at $n={MISSION_N_SHIFT}$")
    ax.set_ylim(0.42, 0.72)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "fig4_profile_shift")

    widths = np.linspace(0.001, 0.020, 10)
    centers = CLASSICAL_PRIOR_SUCCESS
    lower_bounds, upper_bounds, envelope_widths = [], [], []
    for width in widths:
        low_q = np.clip(centers - width, 0.90, 0.9999)
        high_q = np.clip(centers + width, 0.90, 0.9999)
        low, up = imprecise_envelope(counts, BASE_PI, mission_n, low_q, high_q, IMPRECISE_ETA_LOW, IMPRECISE_ETA_HIGH)
        lower_bounds.append(low[0])
        upper_bounds.append(up[0])
        envelope_widths.append(up[0] - low[0])

    fig, ax1 = plt.subplots(figsize=(7.2, 4.7))
    lower_arr = np.array(lower_bounds)
    upper_arr = np.array(upper_bounds)
    autonomous_imp = lower_arr >= R_REQ
    human_imp = (lower_arr < R_REQ) & (upper_arr >= R_REQ)
    ax1.fill_between(widths, 0.50, 0.76, where=autonomous_imp, color="#DFF0D8", alpha=0.40, label="Autonomous-supported range")
    ax1.fill_between(widths, 0.50, 0.76, where=human_imp, color="#FCE8B2", alpha=0.45, label="Human-review range")
    ax1.fill_between(widths, lower_bounds, upper_bounds, color="#B8D8BE", alpha=0.55, label="Reliability bounds")
    ax1.plot(widths, lower_bounds, color="#2F7D32", linewidth=1.8)
    ax1.plot(widths, upper_bounds, color="#2F7D32", linewidth=1.8)
    ax1.axhline(R_REQ, color="#8B1A1A", linewidth=1.2, linestyle=":", label="$R_{req}$")
    imprecision_boundary = first_falls_below(widths, lower_arr, R_REQ)
    if imprecision_boundary is not None:
        ax1.axvline(imprecision_boundary, color="#8B1A1A", linewidth=1.0, linestyle="--")
        ax1.text(imprecision_boundary + 0.0004, R_REQ + 0.006, "review boundary", color="#8B1A1A", fontsize=8)
    ax1.set_xlabel("Prior success-probability half-width")
    ax1.set_ylabel(f"Reliability at $n={MISSION_N_SHIFT}$")
    ax1.set_ylim(0.50, 0.76)
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(widths, envelope_widths, color="#6A3D9A", marker="o", linewidth=2.0, label="Envelope width")
    ax2.set_ylabel("Envelope width")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="lower left")
    save_figure(fig, "fig5_prior_imprecision")

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.plot(n_values, classical_curve, color="#1F77B4", linewidth=2.0, label="Failure-free Bayesian reliability")
    ax.plot(n_values, severity_predictive_curve, color="#C44E52", linewidth=2.0, label="Severity-weighted posterior predictive")
    ax.plot(n_values, severity_plugin_curve, color="#C44E52", linestyle=":", linewidth=1.5, label="Severity-weighted plug-in")
    ax.plot(n_values, accuracy_curve, color="#4D4D4D", linestyle="--", linewidth=1.6, label="Accuracy-only point estimate")
    ax.axhline(R_REQ, color="#8B1A1A", linewidth=1.2, linestyle=":", label="$R_{req}$")
    ax.set_xlabel("Future diagnosis tasks, $n$")
    ax.set_ylabel("Reliability / weighted reliability")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "fig6_severity_weighted")

    profiles = [BASE_PI, STRESS_PI]
    lower_profile, upper_profile = profile_envelope(counts, profiles, n_values)
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    autonomous = lower_profile >= R_REQ
    human_review = (lower_profile < R_REQ) & (upper_profile >= R_REQ)
    restricted = upper_profile < R_REQ
    ax.fill_between(n_values, 0, 1, where=autonomous, color="#DFF0D8", alpha=0.55, label="Autonomous region")
    ax.fill_between(n_values, 0, 1, where=human_review, color="#FCE8B2", alpha=0.55, label="Human-reviewed region")
    ax.fill_between(n_values, 0, 1, where=restricted, color="#F4C7C3", alpha=0.45, label="Restricted region")
    ax.fill_between(n_values, lower_profile, upper_profile, color="#A8C7E6", alpha=0.50, label="Scenario-wise envelope")
    ax.plot(n_values, lower_profile, color="#1F77B4", linewidth=1.8, label="$R_{lower}$")
    ax.plot(n_values, upper_profile, color="#1F77B4", linewidth=1.8, linestyle="--", label="$R_{upper}$")
    ax.axhline(R_REQ, color="#8B1A1A", linewidth=1.4, linestyle=":", label="$R_{req}$")
    ax.set_xlabel("Future diagnosis tasks, $n$")
    ax.set_ylabel("Reliability")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save_figure(fig, "fig7_deployment_regions")

    n_plan = np.array([MISSION_N_PLANNING])
    additional_values = np.arange(
        int(PLANNING_GRID["start"]),
        int(PLANNING_GRID["stop"]) + 1,
        int(PLANNING_GRID["step"]),
    )
    allocations = {
        "Uniform allocation": np.ones(D) / D,
        "Profile-proportional allocation": BASE_PI / BASE_PI.sum(),
        "Risk-targeted allocation": risk_targeted_allocation(counts),
    }
    planning_curves = {}
    for name, allocation in allocations.items():
        values = []
        for total_additional in additional_values:
            planned_counts = add_expected_validation_counts(counts, int(total_additional), allocation)
            low, _ = profile_envelope(planned_counts, profiles, n_plan)
            values.append(low[0])
        planning_curves[name] = np.array(values)

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    colors = ["#4D4D4D", "#1F77B4", "#C44E52"]
    linestyles = ["--", "-.", "-"]
    markers = ["s", "^", "o"]
    crossing_values: dict[str, float | None] = {}
    crossing_linestyles = {
        "Uniform allocation": (0, (3, 2)),
        "Profile-proportional allocation": (0, (6, 2)),
        "Risk-targeted allocation": (0, (1, 1)),
    }
    for (name, values), color, linestyle, marker in zip(planning_curves.items(), colors, linestyles, markers):
        ax.plot(
            additional_values,
            values,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            marker=marker,
            markersize=3.8,
            label=name,
        )
        crossing = first_reaches(additional_values, values, R_REQ)
        crossing_values[name] = crossing
        if crossing is not None:
            ax.axvline(crossing, color=color, linestyle=crossing_linestyles[name], linewidth=1.2, alpha=0.85)
    ax.axhline(R_REQ, color="#8B1A1A", linewidth=1.4, linestyle=":", label="$R_{req}$")
    ax.set_xlabel("Additional expert-adjudicated validation demands")
    ax.set_ylabel(f"$\\widehat{{R}}_{{lower}}$ at $n={MISSION_N_PLANNING}$")
    ax.set_ylim(0.56, 0.72)
    ax.set_xlim(0, 3050)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    save_figure(fig, "fig8_validation_planning")

    lower_n_shift, upper_n_shift = imprecise_envelope(counts, BASE_PI, mission_n)
    lower_profile_shift, upper_profile_shift = profile_envelope(counts, profiles, mission_n)
    unsafe_prob = float(unsafe_upper_probability(counts, profiles))
    unsafe_mission_metrics = {
        f"unsafe_mission_probability_n{n}": float(1.0 - (1.0 - unsafe_prob) ** n)
        for n in UNSAFE_MISSION_LENGTHS
    }
    return {
        "single_demand_accuracy": float(BASE_PI @ observed_success),
        f"reliability_accuracy_n{MISSION_N_SHIFT}": float(point_reliability_curve(observed_success, BASE_PI, mission_n)[0]),
        f"reliability_classical_n{MISSION_N_SHIFT}": float(posterior_reliability_curve(counts, alpha_classical, BASE_PI, mission_n)[0]),
        f"reliability_imprecise_lower_n{MISSION_N_SHIFT}": float(lower_n_shift[0]),
        f"reliability_imprecise_upper_n{MISSION_N_SHIFT}": float(upper_n_shift[0]),
        f"profile_uncertain_lower_n{MISSION_N_SHIFT}": float(lower_profile_shift[0]),
        f"profile_uncertain_upper_n{MISSION_N_SHIFT}": float(upper_profile_shift[0]),
        f"reliability_severity_weighted_plugin_n{MISSION_N_SHIFT}": float(severity_weighted_plugin_curve(counts, alpha_classical, BASE_PI, mission_n)[0]),
        f"reliability_severity_weighted_predictive_n{MISSION_N_SHIFT}": float(severity_weighted_predictive_curve(counts, alpha_classical, BASE_PI, mission_n)[0]),
        "unsafe_upper_probability": unsafe_prob,
        **unsafe_mission_metrics,
        "risk_targeted_allocation_d1": float(allocations["Risk-targeted allocation"][0]),
        "risk_targeted_allocation_d2": float(allocations["Risk-targeted allocation"][1]),
        "risk_targeted_allocation_d3": float(allocations["Risk-targeted allocation"][2]),
        "risk_targeted_allocation_d4": float(allocations["Risk-targeted allocation"][3]),
        "N_add_uniform_first_reaches_R_req": float(crossing_values["Uniform allocation"]) if crossing_values["Uniform allocation"] is not None else math.nan,
        "N_add_profile_first_reaches_R_req": float(crossing_values["Profile-proportional allocation"]) if crossing_values["Profile-proportional allocation"] is not None else math.nan,
        "N_add_risk_first_reaches_R_req": float(crossing_values["Risk-targeted allocation"]) if crossing_values["Risk-targeted allocation"] is not None else math.nan,
    }


def write_generated_validation_counts(counts: np.ndarray) -> None:
    path = DATA_DIR / "generated_validation_counts.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "N", "failure_free", *FAILURE_MODES])
        for d, domain in enumerate(DOMAIN_NAMES):
            writer.writerow([domain, int(N_PER_DOMAIN[d]), *[int(v) for v in counts[d]]])


def write_simulation_parameters() -> None:
    path = DATA_DIR / "simulation_parameters.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "domain_or_index", "value"])
        writer.writerow(["seed", "", SEED])
        writer.writerow(["D", "", D])
        writer.writerow(["M", "", M])
        writer.writerow(["R_req", "", R_REQ])
        writer.writerow(["tau_safe", "", TAU_SAFE])
        writer.writerow(["mission_n_profile_shift", "", MISSION_N_SHIFT])
        writer.writerow(["mission_n_validation_planning", "", MISSION_N_PLANNING])
        writer.writerow(["unsafe_mission_lengths", "", ";".join(map(str, UNSAFE_MISSION_LENGTHS))])
        writer.writerow(["validation_planning_grid", "", f"{PLANNING_GRID['start']}:{PLANNING_GRID['step']}:{PLANNING_GRID['stop']}"])
        writer.writerow(["finite_prior_scenario_set_K_num", "", "q0 vertices from imprecise_q0_low/high crossed with eta_low/eta_high"])
        writer.writerow(["finite_profile_scenario_set_P_num", "", "nominal and stressed operational profiles"])
        for d, domain in enumerate(DOMAIN_NAMES):
            writer.writerow(["task_domain", d + 1, domain])
            writer.writerow(["validation_sample_size_N_d", domain, int(N_PER_DOMAIN[d])])
            writer.writerow(["nominal_profile_pi0", domain, f"{BASE_PI[d]:.8f}"])
            writer.writerow(["stressed_profile_pi1", domain, f"{STRESS_PI[d]:.8f}"])
            writer.writerow(["classical_prior_success", domain, f"{CLASSICAL_PRIOR_SUCCESS[d]:.8f}"])
            writer.writerow(["classical_prior_eta", domain, f"{CLASSICAL_PRIOR_ETA[d]:.8f}"])
            writer.writerow(["imprecise_q0_low", domain, f"{IMPRECISE_Q0_LOW[d]:.8f}"])
            writer.writerow(["imprecise_q0_high", domain, f"{IMPRECISE_Q0_HIGH[d]:.8f}"])
            writer.writerow(["imprecise_eta_low", domain, f"{IMPRECISE_ETA_LOW[d]:.8f}"])
            writer.writerow(["imprecise_eta_high", domain, f"{IMPRECISE_ETA_HIGH[d]:.8f}"])
            for m, label in enumerate(["failure_free", *FAILURE_MODES]):
                writer.writerow([f"true_theta_{m}", domain, f"{TRUE_THETA[d, m]:.8f}"])
        for m, mode in enumerate(FAILURE_MODES):
            writer.writerow(["failure_prior_share", mode, f"{FAILURE_PRIOR_SHARE[m]:.8f}"])
            writer.writerow(["severity_weight", mode, f"{SEVERITY_WEIGHTS[m]:.8f}"])


def write_validation_allocation_summary(metrics: dict[str, float]) -> None:
    path = RESULTS_DIR / "validation_allocation_summary.csv"
    rows = [
        (
            "risk-targeted",
            metrics["N_add_risk_first_reaches_R_req"],
            "Prioritizes high-profile domains with lower or more uncertain failure-free probability; reaches the requirement earliest.",
        ),
        (
            "uniform",
            metrics["N_add_uniform_first_reaches_R_req"],
            "Allocates additional expert-adjudicated validation demands evenly across domains.",
        ),
        (
            "profile-proportional",
            metrics["N_add_profile_first_reaches_R_req"],
            "Allocates validation demands in proportion to the nominal operational profile.",
        ),
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["allocation_strategy", "first_grid_point_reaching_R_req", "interpretation"])
        for strategy, crossing, interpretation in rows:
            writer.writerow([strategy, f"{crossing:.0f}", interpretation])


def write_unsafe_recommendation_summary(metrics: dict[str, float]) -> None:
    path = RESULTS_DIR / "unsafe_recommendation_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", "value", "threshold_used", "deployment_implication"])
        writer.writerow([
            "p_hat_unsafe",
            f"{metrics['unsafe_upper_probability']:.5f}",
            f"single-demand tau_safe={TAU_SAFE:.3f}",
            "Below the single-demand threshold used in the controlled numerical experiments.",
        ])
        for n in UNSAFE_MISSION_LENGTHS:
            key = f"unsafe_mission_probability_n{n}"
            writer.writerow([
                f"p_hat_mission_unsafe({n})",
                f"{metrics[key]:.3f}",
                "mission-level value reported for screening only",
                "Shows how repeated demands can make an at-least-one unsafe-event screen deployment-relevant if a mission-level threshold is specified.",
            ])


def write_summary(counts: np.ndarray, metrics: dict[str, float]) -> None:
    summary_path = RESULTS_DIR / "simulation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", SEED])
        writer.writerow(["R_req", R_REQ])
        writer.writerow(["tau_safe", TAU_SAFE])
        writer.writerow(["mission_n_shift", MISSION_N_SHIFT])
        writer.writerow(["mission_n_planning", MISSION_N_PLANNING])
        writer.writerow([])
        writer.writerow(["domain", "N", "failure_free", *FAILURE_MODES])
        for d, domain in enumerate(DOMAIN_NAMES):
            writer.writerow([domain, int(N_PER_DOMAIN[d]), *[f"{v:.3f}" for v in counts[d]]])
        writer.writerow([])
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, f"{value:.8f}"])


def main() -> None:
    counts = simulate_counts()
    fig1_system_concept()
    fig2_failure_taxonomy()
    metrics = make_numeric_figures(counts)
    write_generated_validation_counts(counts)
    write_simulation_parameters()
    write_summary(counts, metrics)
    write_validation_allocation_summary(metrics)
    write_unsafe_recommendation_summary(metrics)
    print(f"Generated figures in {FIG_DIR}")
    print(f"Wrote generated data to {DATA_DIR}")
    print(f"Wrote summary to {RESULTS_DIR / 'simulation_summary.csv'}")


if __name__ == "__main__":
    main()
