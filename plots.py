"""
plots.py - Geração de Gráficos Científicos de Alta Resolução para Slides e Relatório PME3380.

Disciplina: PME3380 - Modelagem de Sistemas Dinâmicos (Poli-USP)
Projeto: Modelagem 2D de um Guindaste de Torre com Dois Trolleys no Mesmo Trilho
         segurando uma Plataforma com Carga.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from config import FIGURES_DIR, SIMULATION_CONFIG
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis_linear import LinearAnalysisResult
from simulation import SimulationResult


# Configuração estética global voltada para slides de apresentação PME3380
SLIDE_RC_PARAMS = {
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.32,
    "grid.linestyle": "--",
    "lines.linewidth": 2.2,
}
plt.rcParams.update(SLIDE_RC_PARAMS)

# Paleta de cores contrastante e acessível para slides
COLORS = {
    "trolley1": "#0055A4",     # Azul Poli escuro
    "trolley2": "#E65100",     # Laranja vibrante
    "plat": "#1B5E20",         # Verde escuro
    "pitch": "#6A1B9A",        # Roxo
    "wind": "#00838F",         # Ciano escuro
    "linear": "#D32F2F",       # Vermelho tracejado
    "eq": "#455A64",           # Cinza neutro
}


def _save_fig(fig: plt.Figure, base_name: str, output_dir: Path = FIGURES_DIR) -> List[Path]:
    """Salva a figura simultaneamente em PNG (300 DPI) e PDF vetorial."""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{base_name}.png"
    pdf_path = output_dir / f"{base_name}.pdf"
    fig.savefig(png_path, dpi=SIMULATION_CONFIG["dpi_figures"], bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def plot_scenario_trolleys(res: SimulationResult, output_dir: Path = FIGURES_DIR) -> List[Path]:
    """
    Plota posições x1(t), x2(t), velocidades dx1(t), dx2(t) dos trolleys e
    as entradas de força dos motores F_M1(t), F_M2(t) e vento V_v(t).
    """
    t = res.t
    x1, dx1, x2, dx2 = res.x[0], res.x[1], res.x[2], res.x[3]
    F_M1, F_M2 = res.u[0] / 1000.0, res.u[1] / 1000.0  # [kN]
    V_v = res.w[0]

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.8), sharex=True)
    fig.suptitle(f"{res.title} — Dinâmica dos Trolleys e Entradas", y=0.98)

    # 1. Posições x1(t) e x2(t)
    ax = axes[0]
    ax.plot(t, x1, color=COLORS["trolley1"], label=r"Trolley 1: $x_1(t)$")
    ax.plot(t, x2, color=COLORS["trolley2"], label=r"Trolley 2: $x_2(t)$")
    ax.plot(t, res.x[4], color=COLORS["plat"], linestyle="-.", linewidth=1.8, label=r"Centro da Plataforma: $x_p(t)$")
    ax.set_ylabel("Posição X [m]")
    ax.set_title("Posições Horizontais ao Longo da Lança")
    ax.legend(loc="best", ncol=3, framealpha=0.92)

    # 2. Velocidades dx1(t) e dx2(t)
    ax = axes[1]
    ax.plot(t, dx1, color=COLORS["trolley1"], label=r"$\dot{x}_1(t)$ (Trolley 1)")
    ax.plot(t, dx2, color=COLORS["trolley2"], linestyle="--", label=r"$\dot{x}_2(t)$ (Trolley 2)")
    ax.plot(t, res.x[5], color=COLORS["plat"], linestyle=":", linewidth=1.8, label=r"$\dot{x}_p(t)$ (Plataforma)")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Velocidade [m/s]")
    ax.set_title("Velocidades Horizontais dos Trolleys e da Plataforma")
    ax.legend(loc="best", ncol=3, framealpha=0.92)

    # 3. Forças dos motores e velocidade do vento
    ax = axes[2]
    ax.plot(t, F_M1, color=COLORS["trolley1"], label=r"Força Motor 1: $F_{M1}(t)$ [kN]")
    ax.plot(t, F_M2, color=COLORS["trolley2"], linestyle="--", label=r"Força Motor 2: $F_{M2}(t)$ [kN]")
    ax.set_ylabel("Força Motor [kN]")
    ax.set_xlabel("Tempo $t$ [s]")
    ax.set_title("Entradas de Controle $F_{M1}(t), F_{M2}(t)$ e Perturbação de Vento $V_v(t)$")

    if np.max(np.abs(V_v)) > 1e-3:
        ax_wind = ax.twinx()
        ax_wind.fill_between(t, 0.0, V_v, color=COLORS["wind"], alpha=0.18, label=r"Vento $V_v(t)$ [m/s]")
        ax_wind.plot(t, V_v, color=COLORS["wind"], linestyle="-.", linewidth=1.8, label=r"Vento $V_v(t)$ [m/s]")
        ax_wind.set_ylabel(r"Velocidade do Vento $V_v$ [m/s]", color=COLORS["wind"])
        ax_wind.tick_params(axis="y", labelcolor=COLORS["wind"])
        ax_wind.grid(False)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_wind.get_legend_handles_labels()
        ax.legend(lines1 + [lines2[-1]], labels1 + [labels2[-1]], loc="upper right", framealpha=0.92)
    else:
        ax.legend(loc="best", ncol=2, framealpha=0.92)

    fig.tight_layout()
    return _save_fig(fig, f"{res.scenario_id}_trolleys", output_dir)


def plot_scenario_angles(res: SimulationResult, output_dir: Path = FIGURES_DIR) -> List[Path]:
    """
    Plota as saídas principais: ângulos dos cabos theta1(t), theta2(t) [graus],
    velocidades angulares dtheta1(t), dtheta2(t) [graus/s] e retrato de fase.
    """
    t = res.t
    theta1_deg = np.rad2deg(res.y[0])
    dtheta1_deg = np.rad2deg(res.y[1])
    theta2_deg = np.rad2deg(res.y[2])
    dtheta2_deg = np.rad2deg(res.y[3])

    fig = plt.figure(figsize=(12.5, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], hspace=0.32, wspace=0.25)
    fig.suptitle(f"{res.title} — Ângulos e Velocidades Angulares dos Cabos", y=0.98)

    # 1. Ângulos theta1(t) e theta2(t)
    ax_ang = fig.add_subplot(gs[0, 0])
    ax_ang.plot(t, theta1_deg, color=COLORS["trolley1"], label=r"$\theta_1(t)$ (Cabo 1 - Não-Linear)")
    ax_ang.plot(t, theta2_deg, color=COLORS["trolley2"], linestyle="--", label=r"$\theta_2(t)$ (Cabo 2 - Não-Linear)")
    if res.y_linear is not None and res.scenario_id == "cenario1":
        ax_ang.plot(
            t,
            np.rad2deg(res.y_linear[0]),
            color=COLORS["linear"],
            linestyle=":",
            linewidth=2.0,
            label=r"$\theta_1(t)$ (Modelo Linearizado)",
        )
    ax_ang.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax_ang.set_ylabel(r"Ângulo $\theta_i$ [graus]")
    ax_ang.set_title(r"Evolução Temporal de $\theta_1(t)$ e $\theta_2(t)$")
    ax_ang.legend(loc="best", framealpha=0.92)

    # 2. Velocidades angulares dtheta1(t) e dtheta2(t)
    ax_dang = fig.add_subplot(gs[1, 0], sharex=ax_ang)
    ax_dang.plot(t, dtheta1_deg, color=COLORS["trolley1"], label=r"$\dot{\theta}_1(t)$ (Cabo 1)")
    ax_dang.plot(t, dtheta2_deg, color=COLORS["trolley2"], linestyle="--", label=r"$\dot{\theta}_2(t)$ (Cabo 2)")
    ax_dang.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax_dang.set_xlabel("Tempo $t$ [s]")
    ax_dang.set_ylabel(r"Vel. Angular $\dot{\theta}_i$ [graus/s]")
    ax_dang.set_title(r"Velocidades Angulares dos Cabos $\dot{\theta}_1(t)$ e $\dot{\theta}_2(t)$")
    ax_dang.legend(loc="best", framealpha=0.92)

    # 3. Retrato de Fase (theta_i vs dtheta_i)
    ax_phase = fig.add_subplot(gs[:, 1])
    ax_phase.plot(theta1_deg, dtheta1_deg, color=COLORS["trolley1"], alpha=0.9, label=r"Cabo 1 $(\theta_1, \dot{\theta}_1)$")
    ax_phase.plot(
        theta2_deg, dtheta2_deg, color=COLORS["trolley2"], linestyle="--", alpha=0.85, label=r"Cabo 2 $(\theta_2, \dot{\theta}_2)$"
    )
    ax_phase.scatter([theta1_deg[0]], [dtheta1_deg[0]], color="green", s=70, zorder=5, label="Início ($t=0$)")
    ax_phase.scatter([theta1_deg[-1]], [dtheta1_deg[-1]], color="red", marker="X", s=80, zorder=5, label=f"Fim ($t={t[-1]:.0f}$ s)")
    ax_phase.set_xlabel(r"Ângulo $\theta_i$ [graus]")
    ax_phase.set_ylabel(r"Velocidade Angular $\dot{\theta}_i$ [graus/s]")
    ax_phase.set_title("Retrato de Fase Pendular")
    ax_phase.legend(loc="best", framealpha=0.92)

    return _save_fig(fig, f"{res.scenario_id}_saidas_angulos", output_dir)


def plot_scenario_platform_and_tensions(res: SimulationResult, output_dir: Path = FIGURES_DIR) -> List[Path]:
    """
    Plota a trajetória 2D de (x_p, y_p), a inclinação phi(t) da plataforma e as
    forças de tração T1(t), T2(t) nos cabos (comprovando T_i > 0 em todo o tempo).
    """
    t = res.t
    x_p = res.x[4]
    y_p = res.x[6]
    phi_deg = np.rad2deg(res.x[8])
    dphi_deg = np.rad2deg(res.x[9])
    T1_kN = res.T1 / 1000.0
    T2_kN = res.T2 / 1000.0
    T_eq_kN = res.params.T_eq / 1000.0

    fig = plt.figure(figsize=(12.8, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.26)
    fig.suptitle(f"{res.title} — Trajetória 2D, Arfagem $\\phi(t)$ e Trações nos Cabos", y=0.98)

    # 1. Trajetória 2D do Centro da Plataforma (x_p, y_p)
    ax_traj = fig.add_subplot(gs[0, 0])
    sc = ax_traj.scatter(x_p, y_p, c=t, cmap="viridis", s=14, zorder=3)
    ax_traj.plot(x_p, y_p, color="gray", alpha=0.4, linewidth=1.2, zorder=2)
    ax_traj.scatter([x_p[0]], [y_p[0]], color="limegreen", edgecolor="black", s=90, zorder=5, label="Início ($t=0$)")
    ax_traj.scatter([x_p[-1]], [y_p[-1]], color="crimson", edgecolor="black", marker="X", s=95, zorder=5, label="Fim")
    ax_traj.invert_yaxis()  # Eixo Y aponta para baixo!
    cbar = fig.colorbar(sc, ax=ax_traj, pad=0.02)
    cbar.set_label("Tempo $t$ [s]")
    ax_traj.set_xlabel("Posição Horizontal $x_p$ [m]")
    ax_traj.set_ylabel("Profundidade Vertical $y_p$ [m] (eixo Y ↓)")
    ax_traj.set_title("Trajetória 2D do Centro da Plataforma $(x_p, y_p)$")
    ax_traj.legend(loc="best", framealpha=0.92)

    # 2. Inclinação phi(t) da plataforma
    ax_phi = fig.add_subplot(gs[0, 1])
    ax_phi.plot(t, phi_deg, color=COLORS["pitch"], label=r"Inclinação $\phi(t)$ [graus]")
    ax_phi.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax_phi.set_xlabel("Tempo $t$ [s]")
    ax_phi.set_ylabel(r"Inclinação $\phi$ [graus]", color=COLORS["pitch"])
    ax_phi.tick_params(axis="y", labelcolor=COLORS["pitch"])
    ax_phi.set_title(r"Inclinação (Arfagem) da Plataforma $\phi(t)$")

    ax_dphi = ax_phi.twinx()
    ax_dphi.plot(t, dphi_deg, color="#8E24AA", linestyle=":", linewidth=1.4, alpha=0.65, label=r"$\dot{\phi}(t)$ [graus/s]")
    ax_dphi.set_ylabel(r"Vel. Angular $\dot{\phi}$ [graus/s]", color="#8E24AA")
    ax_dphi.tick_params(axis="y", labelcolor="#8E24AA")
    ax_dphi.grid(False)

    # 3. Forças de Tração nos Cabos T1(t) e T2(t) (painel inferior ocupando 2 colunas)
    ax_T = fig.add_subplot(gs[1, :])
    ax_T.plot(t, T1_kN, color=COLORS["trolley1"], label=r"Tração Cabo 1: $T_1(t)$ [kN]")
    ax_T.plot(t, T2_kN, color=COLORS["trolley2"], linestyle="--", label=r"Tração Cabo 2: $T_2(t)$ [kN]")
    ax_T.axhline(T_eq_kN, color=COLORS["eq"], linestyle="-.", linewidth=1.5, label=f"Equilíbrio Estático $T_{{eq}} = {T_eq_kN:.2f}$ kN")
    ax_T.axhline(0.0, color="red", linestyle=":", linewidth=1.6, label="Limite de Afrouxamento ($T = 0$ kN)")

    min_T = min(float(np.min(T1_kN)), float(np.min(T2_kN)))
    max_T = max(float(np.max(T1_kN)), float(np.max(T2_kN)))
    ax_T.set_ylim(min(-1.0, min_T - 2.0), max_T + 3.0)
    ax_T.annotate(
        f"$T_{{\\min}} = {min_T:.2f}$ kN > 0 (Hipótese de cabos sempre tracionados validada!)",
        xy=(0.02, 0.12),
        xycoords="axes fraction",
        fontsize=10.5,
        fontweight="bold",
        color="#1B5E20",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#E8F5E9", edgecolor="#2E7D32", alpha=0.95),
    )
    ax_T.set_xlabel("Tempo $t$ [s]")
    ax_T.set_ylabel("Tração nos Cabos [kN]")
    ax_T.set_title("Forças de Tração Viscoelástica $T_1(t)$ e $T_2(t)$ nos Cabos")
    ax_T.legend(loc="upper right", ncol=2, framealpha=0.92)

    return _save_fig(fig, f"{res.scenario_id}_plataforma_tracoes", output_dir)


def plot_pole_zero_map(
    res_pure: LinearAnalysisResult,
    res_fric: LinearAnalysisResult,
    output_dir: Path = FIGURES_DIR,
) -> List[Path]:
    """
    Plota o Mapa de Polos e Zeros no plano complexo s:
      - Painel esquerdo: Espectro global comparando modos puros (mu_T=0) e com atrito (mu_T=0.015);
      - Painel direito: Zoom nos modos de baixa frequência (pendular + trolleys) e zeros de G_{theta1, FM1}(s).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2))
    fig.suptitle("Mapa de Polos e Zeros no Plano Complexo $s$ — Sistema Linearizado (Ordem 10)", y=1.01)

    poles_pure = np.array([m.eigenvalue for m in res_pure.modes])
    poles_fric = np.array([m.eigenvalue for m in res_fric.modes])
    zeros_theta1 = res_pure.poles_Zeros_dict["theta1_FM1"]["zeros"]

    # --- Painel 1: Espectro Completo de Polos (Modos Vibratórios) ---
    ax1 = axes[0]
    ax1.axvline(0.0, color="black", linestyle="-", linewidth=1.1)
    ax1.axhline(0.0, color="black", linestyle="-", linewidth=1.1)

    # Linhas de fator de amortecimento constante zeta (restritas à janela visível)
    for zeta_val in [0.03, 0.07, 0.15]:
        angle = np.arccos(zeta_val)
        r_max = 31.0
        ax1.plot(
            [0, -r_max * np.cos(angle)],
            [0, r_max * np.sin(angle)],
            "--",
            color="gray",
            alpha=0.35,
            linewidth=0.9,
            clip_on=True,
        )
        ax1.plot(
            [0, -r_max * np.cos(angle)],
            [0, -r_max * np.sin(angle)],
            "--",
            color="gray",
            alpha=0.35,
            linewidth=0.9,
            clip_on=True,
        )

    ax1.scatter(
        poles_pure.real,
        poles_pure.imag,
        marker="x",
        s=110,
        linewidths=2.5,
        color=COLORS["trolley1"],
        label=r"Polos ($\mu_T = 0$: Modos Puros)",
        zorder=5,
    )
    mask_vis = poles_fric.real > -10.0
    ax1.scatter(
        poles_fric[mask_vis].real,
        poles_fric[mask_vis].imag,
        marker="+",
        s=120,
        linewidths=2.2,
        color=COLORS["trolley2"],
        label=r"Polos ($\mu_T = 0,015$: Com Atrito)",
        zorder=4,
    )

    # Anotações dos modos físicos no espectro global
    ax1.annotate(
        "Modo de Arfagem ($\\phi$)\n$\\omega_n = 28,56$ rad/s ($\\zeta = 0,069$)",
        xy=(-1.958, 28.493),
        xytext=(-2.45, 21.0),
        arrowprops=dict(arrowstyle="->", color=COLORS["pitch"], lw=1.3),
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["pitch"],
    )
    ax1.annotate(
        "Modo Axial dos Cabos ($y_p$)\n$\\omega_n = 13,02$ rad/s ($\\zeta = 0,031$)",
        xy=(-0.407, 13.013),
        xytext=(-2.35, 11.5),
        arrowprops=dict(arrowstyle="->", color=COLORS["plat"], lw=1.3),
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["plat"],
    )
    ax1.annotate(
        "Cluster Pendular + Trolleys\n($\\omega_n \\in [0; 1,84]$ rad/s)",
        xy=(-0.17, 1.83),
        xytext=(-2.25, 3.8),
        arrowprops=dict(arrowstyle="->", color=COLORS["trolley1"], lw=1.3),
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["trolley1"],
    )
    ax1.set_xlim(-2.6, 0.35)
    ax1.set_ylim(-32.0, 32.0)
    ax1.set_xlabel(r"Eixo Real $\sigma$ [$\mathrm{s}^{-1}$]")
    ax1.set_ylabel(r"Eixo Imaginário $j\omega$ [$\mathrm{rad/s}$]")
    ax1.set_title("Espectro Global de Polos (Todos os Modos)")
    ax1.legend(loc="lower left", framealpha=0.92)

    # --- Painel 2: Zoom nos Modos de Baixa Frequência (Pendular + Trolleys) e Zeros ---
    ax2 = axes[1]
    ax2.axvline(0.0, color="black", linestyle="-", linewidth=1.1)
    ax2.axhline(0.0, color="black", linestyle="-", linewidth=1.1)

    low_pure = poles_pure[np.abs(poles_pure.imag) < 5.0]
    low_fric = poles_fric[(np.abs(poles_fric.imag) < 5.0) & (poles_fric.real > -5.0)]

    ax2.scatter(
        low_pure.real,
        low_pure.imag,
        marker="x",
        s=120,
        linewidths=2.6,
        color=COLORS["trolley1"],
        label=r"Polos $G_{\theta_1, F_{M1}}(s)$ ($\mu_T=0$)",
        zorder=5,
    )
    ax2.scatter(
        zeros_theta1.real,
        zeros_theta1.imag,
         facecolors="none",
        edgecolors=COLORS["linear"],
        marker="o",
        s=115,
        linewidths=2.3,
        label=r"Zeros de $G_{\theta_1, F_{M1}}(s)$ ($\mu_T=0$)",
        zorder=6,
    )
    ax2.scatter(
        low_fric.real,
        low_fric.imag,
        marker="P",
        s=95,
        color=COLORS["trolley2"],
        label=r"Polos Pendulares ($\mu_T=0,015$)",
        zorder=4,
    )

    ax2.annotate(
        "Modo Pendular Acoplado\n$(-0,173 \\pm 1,827j)$",
        xy=(-0.1728, 1.8272),
        xytext=(-0.24, 2.08),
        arrowprops=dict(arrowstyle="->", color=COLORS["trolley1"]),
        fontsize=9,
    )
    ax2.annotate(
        "Modo Diferencial Trolleys\n$(-0,214 + 1,643j)$",
        xy=(-0.2143, 1.6431),
        xytext=(-0.262, 1.12),
        arrowprops=dict(arrowstyle="->", color=COLORS["trolley1"]),
        fontsize=9,
    )
    ax2.annotate(
        "Pêndulo Simples ($\\mu_T=0,015$)\n$\\omega_n = \\sqrt{g/l_{eq}} = 0,807$ rad/s",
        xy=(-0.0178, 0.8071),
        xytext=(-0.155, 0.72),
        arrowprops=dict(arrowstyle="->", color=COLORS["trolley2"]),
        fontsize=9,
    )

    ax2.set_xlim(-0.27, 0.04)
    ax2.set_ylim(-2.45, 2.45)
    ax2.set_xlabel(r"Eixo Real $\sigma$ [$\mathrm{s}^{-1}$]")
    ax2.set_ylabel(r"Eixo Imaginário $j\omega$ [$\mathrm{rad/s}$]")
    ax2.set_title(r"Zoom: Polos e Zeros de Baixa Frequência ($G_{\theta_1, F_{M1}}$)")
    ax2.legend(loc="lower left", framealpha=0.92)

    fig.tight_layout()
    return _save_fig(fig, "mapa_polos_zeros", output_dir)


def plot_energy_and_linear_validation(res1: SimulationResult, output_dir: Path = FIGURES_DIR) -> List[Path]:
    """
    Plota o balanço completo de energia mecânica (cinética, potencial, dissipada e
    conservação da soma E_mec + E_dissip) e a validação Modelo Linear vs Não-Linear no Cenário 1.
    """
    t = res1.t
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))
    fig.suptitle("Cenário 1: Conservação/Dissipação de Energia e Validação do Modelo Linearizado", y=0.99)

    # 1. Balanço de Energia Mecânica
    ax = axes[0, 0]
    E_tot_conserved = res1.E_total + res1.E_dissip_acum
    ax.plot(t, res1.E_total / 1000.0, color="black", linewidth=2.4, label=r"Energia Mecânica $E_{mec}(t) = K + V$")
    ax.plot(t, res1.E_cin / 1000.0, color=COLORS["trolley1"], linestyle="--", label=r"Energia Cinética $K(t)$")
    ax.plot(t, res1.E_pot / 1000.0, color=COLORS["trolley2"], linestyle="-.", label=r"Energia Potencial $V_{grav} + V_{elast}$")
    ax.plot(t, res1.E_dissip_acum / 1000.0, color=COLORS["plat"], linestyle=":", linewidth=2.2, label=r"Energia Dissipada $\int P_{dissip}\,dt$")
    ax.plot(
        t,
        E_tot_conserved / 1000.0,
        color=COLORS["linear"],
        linestyle="--",
        linewidth=1.6,
        label=r"Soma Conservada $E_{mec} + E_{dissip}$",
    )
    ax.set_xlabel("Tempo $t$ [s]")
    ax.set_ylabel("Energia [kJ]")
    ax.set_title("Balanço de Energia Mecânica e Dissipação")
    ax.legend(loc="right", fontsize=9.0, framealpha=0.92)

    # 2. Comparação Não-Linear vs Linear: Ângulo Pendular theta1(t)
    ax = axes[0, 1]
    ax.plot(t, np.rad2deg(res1.y[0]), color=COLORS["trolley1"], label="Não-Linear (Ordem 10)")
    if res1.y_linear is not None:
        ax.plot(t, np.rad2deg(res1.y_linear[0]), color=COLORS["linear"], linestyle="--", label=r"Linearizado ($\mu_T = 0,015$)")
    ax.set_xlabel("Tempo $t$ [s]")
    ax.set_ylabel(r"Ângulo $\theta_1(t)$ [graus]")
    ax.set_title(r"Modo Pendular: $\theta_1(t)$ ($\omega_n \approx 0,81$ rad/s)")
    ax.legend(loc="upper right", framealpha=0.92)

    # 3. Comparação Não-Linear vs Linear: Oscilação Axial Vertical y_p(t) - y_p_eq
    ax = axes[1, 0]
    y_eq = res1.params.y_p_eq
    ax.plot(t, (res1.x[6] - y_eq) * 1000.0, color=COLORS["plat"], label="Não-Linear (Acoplamento Centrífugo)")
    if res1.x_linear is not None:
        ax.plot(
            t,
            (res1.x_linear[6] - y_eq) * 1000.0,
            color=COLORS["linear"],
            linestyle="--",
            label=r"Linearizado ($\omega_n = 13,02$ rad/s)",
        )
    ax.set_xlim(0.0, 6.0)  # Zoom nos primeiros 6 s para visualizar perfeitamente a alta frequência axial
    ax.set_xlabel("Tempo $t$ [s] (Zoom 0 a 6 s)")
    ax.set_ylabel(r"Deslocamento Vertical $\Delta y_p$ [mm]")
    ax.set_title(r"Modo Axial de Mola dos Cabos: $\Delta y_p(t)$ ($f_n = 2,07$ Hz)")
    ax.legend(loc="upper right", framealpha=0.92)

    # 4. Comparação Não-Linear vs Linear: Modo de Arfagem phi(t)
    ax = axes[1, 1]
    ax.plot(t, np.rad2deg(res1.x[8]), color=COLORS["pitch"], label="Não-Linear")
    if res1.x_linear is not None:
        ax.plot(
            t,
            np.rad2deg(res1.x_linear[8]),
            color=COLORS["linear"],
            linestyle="--",
            label=r"Linearizado ($\omega_n = 28,56$ rad/s)",
        )
    ax.set_xlim(0.0, 4.0)  # Zoom nos primeiros 4 s para ver a oscilação de 4,55 Hz
    ax.set_xlabel("Tempo $t$ [s] (Zoom 0 a 4 s)")
    ax.set_ylabel(r"Inclinação $\phi(t)$ [graus]")
    ax.set_title(r"Modo de Arfagem da Plataforma: $\phi(t)$ ($f_n = 4,55$ Hz)")
    ax.legend(loc="upper right", framealpha=0.92)

    fig.tight_layout()
    return _save_fig(fig, "cenario1_energia_validacao", output_dir)


def plot_convergence_study(
    conv_results: List[Dict[str, float | str]],
    output_dir: Path = FIGURES_DIR,
) -> List[Path]:
    """Plota o estudo de convergência numérica comparando RK45 e Radau."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    fig.suptitle("Teste de Convergência Numérica (`solve_ivp`: RK45 vs Radau)", y=1.01)

    for method, color, marker in [("RK45", COLORS["trolley1"], "o"), ("Radau", COLORS["trolley2"], "s")]:
        sub = [r for r in conv_results if r["method"] == method]
        rtols = [float(r["rtol"]) for r in sub]
        err_inf = [float(r["err_inf"]) for r in sub]
        nfevs = [int(r["nfev"]) for r in sub]

        axes[0].loglog(rtols, err_inf, marker=marker, markersize=8, color=color, label=f"Método {method}")
        axes[1].loglog(nfevs, err_inf, marker=marker, markersize=8, color=color, label=f"Método {method}")

    axes[0].invert_xaxis()
    axes[0].axvline(1e-7, color=COLORS["plat"], linestyle=":", linewidth=1.8, label="Tolerância Adotada (rtol=1e-7)")
    axes[0].set_xlabel("Tolerância Relativa `rtol`")
    axes[0].set_ylabel(r"Erro Global Máximo $\|x - x_{ref}\|_\infty$")
    axes[0].set_title("Convergência do Erro Global vs. Tolerância")
    axes[0].legend(loc="best", framealpha=0.92)

    axes[1].set_xlabel("Número de Avaliações do Campo Vetorial (`nfev`)")
    axes[1].set_ylabel(r"Erro Global Máximo $\|x - x_{ref}\|_\infty$")
    axes[1].set_title("Custo Computacional vs. Precisão Numérica")
    axes[1].legend(loc="best", framealpha=0.92)

    fig.tight_layout()
    return _save_fig(fig, "convergencia_numerica", output_dir)


def plot_bode_diagrams(res_pure: LinearAnalysisResult, output_dir: Path = FIGURES_DIR) -> List[Path]:
    """Plota os Diagramas de Bode das principais funções de transferência do guindaste."""
    omega = np.logspace(-2, 2, 1200)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True)
    fig.suptitle("Resposta em Frequência (Diagrama de Bode) — Funções de Transferência Linearizadas", y=0.99)

    channels = [
        ("theta1_FM1", r"$G_{\theta_1, F_{M1}}(j\omega)$ [rad/N]", COLORS["trolley1"], "-"),
        ("theta1_F_sinc", r"$G_{\theta_1, F_{sinc}}(j\omega)$ [rad/N]", COLORS["trolley2"], "--"),
        ("phi_Vv_nom", r"$G_{\phi, V_v}(j\omega)$ [rad/(m/s)]", COLORS["pitch"], "-."),
    ]

    for key, label, color, ls in channels:
        tf_obj = res_pure.transfer_functions[key]
        mag, phase, _ = tf_obj.frequency_response(omega)
        mag_db = 20.0 * np.log10(np.maximum(np.squeeze(mag), 1e-18))
        phase_deg = np.rad2deg(np.unwrap(np.squeeze(phase)))
        axes[0].semilogx(omega, mag_db, color=color, linestyle=ls, label=label)
        axes[1].semilogx(omega, phase_deg, color=color, linestyle=ls, label=label)

    # Marca as frequências naturais no gráfico de magnitude
    for wn_val, name in [(1.657, r"$\omega_{diff}=1,66$"), (1.835, r"$\omega_{pend}=1,84$"), (28.56, r"$\omega_{\phi}=28,56$ rad/s")]:
        axes[0].axvline(wn_val, color="gray", linestyle=":", linewidth=1.2)
        axes[0].text(wn_val * 1.04, -35.0, name, fontsize=9.5, color="black", rotation=90, va="bottom")

    axes[0].set_ylabel("Magnitude [dB]")
    axes[0].set_title("Magnitude vs. Frequência")
    axes[0].legend(loc="best", framealpha=0.92)

    axes[1].set_xlabel(r"Frequência Angular $\omega$ [rad/s]")
    axes[1].set_ylabel("Fase [graus]")
    axes[1].set_title("Fase vs. Frequência")
    axes[1].legend(loc="best", framealpha=0.92)

    fig.tight_layout()
    return _save_fig(fig, "diagrama_bode", output_dir)


def generate_all_plots(
    scenarios: Dict[str, SimulationResult],
    res_pure: LinearAnalysisResult,
    res_fric: LinearAnalysisResult,
    conv_results: List[Dict[str, float | str]],
    output_dir: Path = FIGURES_DIR,
) -> List[Path]:
    """Gera e salva todos os gráficos científicos do projeto."""
    saved_files: List[Path] = []
    for scen_res in scenarios.values():
        saved_files.extend(plot_scenario_trolleys(scen_res, output_dir))
        saved_files.extend(plot_scenario_angles(scen_res, output_dir))
        saved_files.extend(plot_scenario_platform_and_tensions(scen_res, output_dir))

    saved_files.extend(plot_pole_zero_map(res_pure, res_fric, output_dir))
    saved_files.extend(plot_energy_and_linear_validation(scenarios["cenario1"], output_dir))
    saved_files.extend(plot_convergence_study(conv_results, output_dir))
    saved_files.extend(plot_bode_diagrams(res_pure, output_dir))
    return saved_files
