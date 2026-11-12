"""
animation.py - Animação 2D Visualmente Rica do Guindaste de Torre com Dois Trolleys.

Disciplina: PME3380 - Modelagem de Sistemas Dinâmicos (Poli-USP)
Projeto: Modelagem 2D de um Guindaste de Torre com Dois Trolleys no Mesmo Trilho
         segurando uma Plataforma com Carga.

Elementos visuais físicos representados:
  - Torre vertical treliçada em x = 0, cabine do operador, ápice (cathead) e tirantes;
  - Lança horizontal treliçada (trilho em y = 0 até L_total = 50 m) e contra-lança com contrapeso;
  - 2 trolleys com rodinhas apoiadas no trilho em x1(t) e x2(t);
  - 2 cabos viscoelásticos ligando os trolleys às extremidades A e B da plataforma,
    com cor e espessura mapeadas dinamicamente pela tração mecânica T1(t), T2(t);
  - Plataforma rígida inclinada por phi(t) e bloco da carga (m_c) apoiado e rotacionando junto;
  - Setas dinâmicas em tempo real das forças dos motores F_M1(t), F_M2(t) e das rajadas de vento V_v(t);
  - Rastro pontilhado da trajetória do centro de massa e painel HUD de telemetria;
  - Painel lateral sincronizado mostrando a evolução temporal de theta1(t), theta2(t), phi(t) e T1(t), T2(t).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from config import ANIMATIONS_DIR, FIGURES_DIR
import matplotlib
import matplotlib.animation as mpl_anim
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from simulation import SimulationResult


def _draw_static_crane_structure(ax: plt.Axes, L_total: float, y_ground: float = 24.0) -> None:
    """
    Desenha a estrutura metálica treliçada completa do guindaste de torre:
    fundação, torre vertical treliçada, contra-lança com blocos de contrapeso,
    lança principal treliçada, trilho horizontal em y = 0, cabine e tirantes.
    """
    steel_dark = "#263238"
    steel_mid = "#455A64"
    crane_yellow = "#F59E0B"
    crane_orange = "#D97706"

    # Solo e fundação de concreto
    ax.axhline(y_ground, color="#3E2723", linewidth=3.0, zorder=1)
    ax.fill_between([-18.0, L_total + 5.0], y_ground, y_ground + 2.5, color="#D7CCC8", alpha=0.65, zorder=1)
    foundation = patches.Rectangle((-3.2, y_ground - 1.2), 4.6, 1.2, facecolor="#78909C", edgecolor=steel_dark, lw=1.8, zorder=2)
    ax.add_patch(foundation)

    # Torre vertical treliçada (x in [-1.8, 0.0], de y = y_ground até y = -5.5 m)
    x_left, x_right = -1.8, 0.0
    ax.plot([x_left, x_left], [-5.2, y_ground - 1.2], color=crane_orange, lw=3.0, zorder=2)
    ax.plot([x_right, x_right], [-5.2, y_ground - 1.2], color=crane_orange, lw=3.0, zorder=2)

    y_nodes_tower = np.arange(-1.5, y_ground - 1.0, 2.2)
    for k in range(len(y_nodes_tower) - 1):
        y0, y1 = y_nodes_tower[k], y_nodes_tower[k + 1]
        ax.plot([x_left, x_right], [y0, y0], color=crane_yellow, lw=1.3, zorder=2)
        ax.plot([x_left, x_right], [y0, y1], color=crane_yellow, lw=1.1, zorder=2)
        ax.plot([x_right, x_left], [y0, y1], color=crane_yellow, lw=1.1, zorder=2)

    # Ápice da torre (cathead) de y = -1.6 m até y = -5.5 m
    apex_x = [x_left, -0.9, x_right]
    apex_y = [-1.6, -5.5, -1.6]
    ax.plot(apex_x, apex_y, color=crane_orange, lw=2.8, zorder=3)

    # Lança horizontal treliçada (de x = 0 até L_total)
    # Banzo inferior = trilho em y = 0; banzo superior em y = -1.6 m
    y_rail = 0.0
    y_top = -1.6
    ax.plot([0.0, L_total], [y_top, y_top], color=crane_orange, lw=2.5, zorder=2)
    x_nodes_jib = np.linspace(0.0, L_total, 26)
    for k in range(len(x_nodes_jib) - 1):
        xa, xb = x_nodes_jib[k], x_nodes_jib[k + 1]
        xm = 0.5 * (xa + xb)
        ax.plot([xa, xm, xb], [y_rail, y_top, y_rail], color=crane_yellow, lw=1.2, zorder=2)
        ax.plot([xa, xa], [y_rail, y_top], color=crane_orange, lw=1.0, alpha=0.7, zorder=2)

    # Contra-lança treliçada (de x = -13.5 m até x = -1.8 m)
    x_cj_start = -13.5
    ax.plot([x_cj_start, x_left], [y_rail, y_rail], color=crane_orange, lw=2.6, zorder=2)
    ax.plot([x_cj_start, x_left], [y_top, y_top], color=crane_orange, lw=2.2, zorder=2)
    x_nodes_cj = np.linspace(x_cj_start, x_left, 7)
    for k in range(len(x_nodes_cj) - 1):
        xa, xb = x_nodes_cj[k], x_nodes_cj[k + 1]
        xm = 0.5 * (xa + xb)
        ax.plot([xa, xm, xb], [y_rail, y_top, y_rail], color=crane_yellow, lw=1.2, zorder=2)

    # Blocos de contrapeso na extremidade da contra-lança
    for idx_cw, x_cw in enumerate([-13.0, -11.9, -10.8]):
        cw_rect = patches.FancyBboxPatch(
            (x_cw, -0.2),
            0.9,
            2.6,
            boxstyle="round,pad=0.05",
            facecolor="#546E7A" if idx_cw % 2 == 0 else "#607D8B",
            edgecolor=steel_dark,
            lw=1.4,
            zorder=3,
        )
        ax.add_patch(cw_rect)
    ax.text(-11.4, 3.2, "Contrapeso", fontsize=8.5, ha="center", color=steel_dark, fontweight="bold")

    # Tirantes metálicos de suspensão (do ápice em (-0.9, -5.5) para a lança e contra-lança)
    for x_tie in [14.0, 32.0, 46.0, -12.0]:
        ax.plot([-0.9, x_tie], [-5.5, y_top], color=steel_mid, lw=1.4, linestyle="-", zorder=2)

    # Cabine do operador no topo da torre
    cabin = patches.FancyBboxPatch(
        (-0.3, -1.3),
        1.4,
        1.8,
        boxstyle="round,pad=0.08",
        facecolor="#ECEFF1",
        edgecolor=steel_dark,
        lw=1.6,
        zorder=4,
    )
    window = patches.Rectangle((0.35, -1.05), 0.65, 0.9, facecolor="#81D4FA", edgecolor=steel_dark, lw=1.0, zorder=5)
    ax.add_patch(cabin)
    ax.add_patch(window)

    # Trilho principal (em y = 0)
    ax.plot([0.0, L_total], [0.0, 0.0], color=steel_dark, lw=3.6, zorder=4, label="Trilho da Lança (y = 0)")


def _compute_polygon_corners(cx: float, cy: float, width: float, height: float, phi: float) -> np.ndarray:
    """
    Calcula os 4 vértices rotacionados de um retângulo centrado em (cx, cy) com
    inclinação phi (lembrando que o eixo X aponta para a direita e Y aponta para baixo).
    """
    c, s = np.cos(phi), np.sin(phi)
    hw, hh = 0.5 * width, 0.5 * height
    local_pts = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=float)
    rot_mat = np.array([[c, -s], [s, c]], dtype=float)
    return (local_pts @ rot_mat.T) + np.array([cx, cy])


def create_crane_animation(
    res: SimulationResult,
    output_dir: Path = ANIMATIONS_DIR,
    fps: int = 30,
    frame_stride: int = 3,
    save_mp4: bool = True,
    save_gif: bool = True,
    show_interactive: bool = False,
) -> Dict[str, Path]:
    """
    Cria a animação 2D completa do guindaste com painel lateral sincronizado e
    exporta em `.mp4`, `.gif` e um snapshot estático de alta resolução (`.png`).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    p = res.params

    # Subamostragem de quadros para manter a renderização rápida e fluida
    indices = np.arange(0, len(res.t), frame_stride)
    if indices[-1] != len(res.t) - 1:
        indices = np.append(indices, len(res.t) - 1)

    t_frames = res.t[indices]
    n_frames = len(indices)

    # Configuração da figura: Painel físico à esquerda (maior) + 2 gráficos sincronizados à direita
    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1.0], height_ratios=[1.0, 1.0], wspace=0.22, hspace=0.28)
    fig.suptitle(f"PME3380 (Poli-USP) — {res.title}", fontsize=14, fontweight="bold", y=0.97)

    ax_crane = fig.add_subplot(gs[:, 0])
    ax_theta = fig.add_subplot(gs[0, 1])
    ax_tens = fig.add_subplot(gs[1, 1], sharex=ax_theta)

    # --- Configuração do Painel Físico do Guindaste ---
    y_ground = max(23.5, float(np.max(res.x[6])) + 6.5)
    _draw_static_crane_structure(ax_crane, p.L_total, y_ground=y_ground)

    # Limites de visualização mantendo proporção geométrica física (aspect = equal)
    x_min_view = -14.5
    x_max_view = p.L_total + 3.0
    ax_crane.set_xlim(x_min_view, x_max_view)
    ax_crane.set_ylim(y_ground + 2.0, -13.5)  # Invertido: Y positivo aponta para baixo!
    ax_crane.set_aspect("equal", adjustable="box")
    ax_crane.set_xlabel("Posição Horizontal $X$ ao longo da Lança [m]")
    ax_crane.set_ylabel("Coordenada Vertical $Y$ [m] (Sentido da Gravidade $\\downarrow$)")
    ax_crane.set_title("Visualização Física Dinâmica 2D (Escala Real)")

    # Mapa de cores para a tensão mecânica nos cabos T1, T2 [kN]
    T_min_kN = max(0.0, float(min(np.min(res.T1), np.min(res.T2))) / 1000.0 - 1.5)
    T_max_kN = float(max(np.max(res.T1), np.max(res.T2))) / 1000.0 + 1.5
    norm_T = mcolors.Normalize(vmin=T_min_kN, vmax=T_max_kN)
    cmap_T = plt.get_cmap("plasma")
    sm_T = plt.cm.ScalarMappable(cmap=cmap_T, norm=norm_T)
    sm_T.set_array([])
    cbar = fig.colorbar(sm_T, ax=ax_crane, orientation="horizontal", fraction=0.038, pad=0.09, aspect=35)
    cbar.set_label("Tração Mecânica nos Cabos $T_1(t), T_2(t)$ [kN]", fontsize=10, fontweight="bold")

    # Elementos móveis: Trolleys 1 e 2 + 4 rodinhas
    trolley_w, trolley_h = 1.7, 0.65
    trolley1_box = patches.FancyBboxPatch(
        (0, 0), trolley_w, trolley_h, boxstyle="round,pad=0.05", facecolor="#0055A4", edgecolor="#0D253F", lw=1.6, zorder=6
    )
    trolley2_box = patches.FancyBboxPatch(
        (0, 0), trolley_w, trolley_h, boxstyle="round,pad=0.05", facecolor="#E65100", edgecolor="#4E1A00", lw=1.6, zorder=6
    )
    ax_crane.add_patch(trolley1_box)
    ax_crane.add_patch(trolley2_box)

    wheel_radius = 0.20
    wheels = [
        patches.Circle((0, -wheel_radius), wheel_radius, facecolor="#263238", edgecolor="white", lw=1.0, zorder=7)
        for _ in range(4)
    ]
    for w_patch in wheels:
        ax_crane.add_patch(w_patch)

    # Cabos 1 e 2 + Pinos de ancoragem
    (cable1_line,) = ax_crane.plot([], [], lw=3.2, zorder=5)
    (cable2_line,) = ax_crane.plot([], [], lw=3.2, zorder=5)
    (anchors_scatter,) = ax_crane.plot([], [], "o", color="#263238", markersize=6, zorder=8)

    # Plataforma rígida (m_p) e Bloco da Carga (m_c)
    plat_thickness = 0.42
    h_c = p.h_c
    plat_poly = patches.Polygon(np.zeros((4, 2)), closed=True, facecolor="#2E7D32", edgecolor="#1B5E20", lw=1.8, zorder=7)
    load_poly = patches.Polygon(np.zeros((4, 2)), closed=True, facecolor="#1565C0", edgecolor="#0D47A1", lw=1.8, alpha=0.90, zorder=7)
    ax_crane.add_patch(plat_poly)
    ax_crane.add_patch(load_poly)

    # Marcador do Centro de Massa (CG) e rastro pontilhado
    (trail_line,) = ax_crane.plot([], [], ":", color="#1B5E20", lw=1.8, alpha=0.85, zorder=4, label="Trajetória $(x_p, y_p)$")
    (cg_marker,) = ax_crane.plot([], [], "P", color="#FFD600", markeredgecolor="black", markersize=9, zorder=9, label="CG Composto")

    # Vetores dinâmicos (Quivers): Forças dos motores F_M1, F_M2 e Vento V_v
    # Usamos setas FancyArrow / linhas com cabeça para controle preciso em coordenadas de dados
    (fm1_arrow,) = ax_crane.plot([], [], color="#003380", lw=3.0, marker=">", markevery=[-1], markersize=8, zorder=9)
    (fm2_arrow,) = ax_crane.plot([], [], color="#BF360C", lw=3.0, marker=">", markevery=[-1], markersize=8, zorder=9)
    fm1_text = ax_crane.text(0, 0, "", fontsize=8.5, fontweight="bold", color="#003380", ha="center", zorder=10)
    fm2_text = ax_crane.text(0, 0, "", fontsize=8.5, fontweight="bold", color="#BF360C", ha="center", zorder=10)

    # Linhas de fluxo / setas de rajada de vento incidindo na lateral da carga
    n_wind_arrows = 4
    wind_arrows = [
        ax_crane.plot([], [], color="#00838F", lw=2.2, marker=">", markevery=[-1], markersize=7, alpha=0.85, zorder=8)[0]
        for _ in range(n_wind_arrows)
    ]
    wind_label = ax_crane.text(0, 0, "", fontsize=9.0, fontweight="bold", color="#006064", ha="right", zorder=10)

    # Painel de Telemetria (HUD) no canto superior direito da área do guindaste
    hud_text = ax_crane.text(
        0.985,
        0.975,
        "",
        transform=ax_crane.transAxes,
        fontsize=9.2,
        fontfamily="monospace",
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FAFAFA", edgecolor="#37474F", alpha=0.93, lw=1.3),
        zorder=12,
    )

    # --- Configuração do Painel Lateral Sincronizado ---
    t_all = res.t
    th1_deg_all = np.rad2deg(res.y[0])
    th2_deg_all = np.rad2deg(res.y[2])
    phi_deg_all = np.rad2deg(res.x[8])
    T1_kN_all = res.T1 / 1000.0
    T2_kN_all = res.T2 / 1000.0

    # Gráfico Superior Direito: theta1(t), theta2(t) e phi(t)
    ax_theta.plot(t_all, th1_deg_all, color="#0055A4", alpha=0.22, lw=1.2)
    ax_theta.plot(t_all, th2_deg_all, color="#E65100", alpha=0.22, lw=1.2, ls="--")
    ax_theta.plot(t_all, phi_deg_all, color="#6A1B9A", alpha=0.22, lw=1.2, ls=":")
    (line_th1_dyn,) = ax_theta.plot([], [], color="#0055A4", lw=2.2, label=r"$\theta_1(t)$ [°]")
    (line_th2_dyn,) = ax_theta.plot([], [], color="#E65100", lw=2.2, ls="--", label=r"$\theta_2(t)$ [°]")
    (line_phi_dyn,) = ax_theta.plot([], [], color="#6A1B9A", lw=2.0, ls="-.", label=r"$\phi(t)$ [°]")
    cursor_theta = ax_theta.axvline(0.0, color="crimson", lw=1.4, ls=":")
    ax_theta.set_xlim(float(t_all[0]), float(t_all[-1]))
    y_ang_lim = max(1.0, float(np.max(np.abs(np.concatenate([th1_deg_all, th2_deg_all, phi_deg_all])))) * 1.18)
    ax_theta.set_ylim(-y_ang_lim, y_ang_lim)
    ax_theta.set_ylabel("Ângulo [graus]")
    ax_theta.set_title(r"Evolução Sincronizada: $\theta_1(t), \theta_2(t)$ e $\phi(t)$")
    ax_theta.legend(loc="upper right", ncol=3, fontsize=8.8, framealpha=0.92)

    # Gráfico Inferior Direito: Trações T1(t) e T2(t) [kN]
    ax_tens.plot(t_all, T1_kN_all, color="#0055A4", alpha=0.22, lw=1.2)
    ax_tens.plot(t_all, T2_kN_all, color="#E65100", alpha=0.22, lw=1.2, ls="--")
    ax_tens.axhline(p.T_eq / 1000.0, color="#455A64", ls=":", lw=1.2, label=r"$T_{eq}$")
    (line_T1_dyn,) = ax_tens.plot([], [], color="#0055A4", lw=2.2, label=r"$T_1(t)$ [kN]")
    (line_T2_dyn,) = ax_tens.plot([], [], color="#E65100", lw=2.2, ls="--", label=r"$T_2(t)$ [kN]")
    cursor_tens = ax_tens.axvline(0.0, color="crimson", lw=1.4, ls=":")
    ax_tens.set_ylim(T_min_kN, T_max_kN)
    ax_tens.set_xlabel("Tempo $t$ [s]")
    ax_tens.set_ylabel("Tração [kN]")
    ax_tens.set_title(r"Evolução Sincronizada: Trações $T_1(t)$ e $T_2(t)$")
    ax_tens.legend(loc="upper right", ncol=3, fontsize=8.8, framealpha=0.92)

    def update_frame(frame_idx: int):
        k = int(indices[frame_idx])
        tk = float(res.t[k])
        x1 = float(res.x[0, k])
        x2 = float(res.x[2, k])
        x_p = float(res.x[4, k])
        y_p = float(res.x[6, k])
        phi = float(res.x[8, k])

        xA = float(res.x_A[k])
        yA = float(res.y_A[k])
        xB = float(res.x_B[k])
        yB = float(res.y_B[k])
        T1_kN = float(res.T1[k]) / 1000.0
        T2_kN = float(res.T2[k]) / 1000.0
        FM1 = float(res.u[0, k])
        FM2 = float(res.u[1, k])
        Vv = float(res.w[0, k])
        F_wind = float(res.F_vento[k])
        th1_deg = float(np.rad2deg(res.y[0, k]))
        th2_deg = float(np.rad2deg(res.y[2, k]))
        phi_deg = float(np.rad2deg(phi))

        # 1. Atualiza posição dos Trolleys e de suas 4 rodinhas no trilho (y = 0)
        trolley1_box.set_bounds(x1 - 0.5 * trolley_w, -trolley_h - 2 * wheel_radius, trolley_w, trolley_h)
        trolley2_box.set_bounds(x2 - 0.5 * trolley_w, -trolley_h - 2 * wheel_radius, trolley_w, trolley_h)
        wheels[0].center = (x1 - 0.5, -wheel_radius)
        wheels[1].center = (x1 + 0.5, -wheel_radius)
        wheels[2].center = (x2 - 0.5, -wheel_radius)
        wheels[3].center = (x2 + 0.5, -wheel_radius)

        # 2. Atualiza Cabos 1 e 2 com cor e espessura proporcionais à tração T1, T2
        cable1_line.set_data([x1, xA], [0.0, yA])
        cable2_line.set_data([x2, xB], [0.0, yB])
        cable1_line.set_color(cmap_T(norm_T(T1_kN)))
        cable2_line.set_color(cmap_T(norm_T(T2_kN)))
        cable1_line.set_linewidth(2.4 + 2.0 * np.clip((T1_kN - T_min_kN) / max(1e-3, T_max_kN - T_min_kN), 0.0, 1.0))
        cable2_line.set_linewidth(2.4 + 2.0 * np.clip((T2_kN - T_min_kN) / max(1e-3, T_max_kN - T_min_kN), 0.0, 1.0))
        anchors_scatter.set_data([x1, x2, xA, xB], [0.0, 0.0, yA, yB])

        # 3. Atualiza Plataforma rígida e Bloco da Carga apoiado sobre ela
        plat_corners = _compute_polygon_corners(x_p, y_p, p.L_p, plat_thickness, phi)
        plat_poly.set_xy(plat_corners)

        # O centro da carga fica a uma distância (plat_thickness/2 + h_c/2) "acima" da plataforma
        # Como o eixo Y aponta para baixo, a normal "para cima" da plataforma é (sin(phi), -cos(phi))
        offset_normal = 0.5 * plat_thickness + 0.5 * h_c
        cx_load = x_p + offset_normal * np.sin(phi)
        cy_load = y_p - offset_normal * np.cos(phi)
        load_corners = _compute_polygon_corners(cx_load, cy_load, p.L_c, h_c, phi)
        load_poly.set_xy(load_corners)

        # Centro de massa composto e rastro pontilhado
        cg_marker.set_data([x_p], [y_p - 0.85 * offset_normal])
        trail_line.set_data(res.x[4, : k + 1], res.x[6, : k + 1])

        # 4. Setas dinâmicas de força dos motores F_M1(t) e F_M2(t)
        scale_F = 0.0014  # m por N (~3.6 m para 2600 N)
        if abs(FM1) > 15.0:
            dx_f1 = np.sign(FM1) * max(1.2, abs(FM1) * scale_F)
            y_f1 = -1.25
            fm1_arrow.set_data([x1, x1 + dx_f1], [y_f1, y_f1])
            fm1_arrow.set_marker(">" if FM1 > 0 else "<")
            fm1_text.set_position((x1 + 0.5 * dx_f1, y_f1 - 0.55))
            fm1_text.set_text(f"$F_{{M1}}={FM1/1000:+.2f}$ kN")
        else:
            fm1_arrow.set_data([], [])
            fm1_text.set_text("")

        if abs(FM2) > 15.0:
            dx_f2 = np.sign(FM2) * max(1.2, abs(FM2) * scale_F)
            y_f2 = -2.25
            fm2_arrow.set_data([x2, x2 + dx_f2], [y_f2, y_f2])
            fm2_arrow.set_marker(">" if FM2 > 0 else "<")
            fm2_text.set_position((x2 + 0.5 * dx_f2, y_f2 - 0.55))
            fm2_text.set_text(f"$F_{{M2}}={FM2/1000:+.2f}$ kN")
        else:
            fm2_arrow.set_data([], [])
            fm2_text.set_text("")

        # 5. Setas dinâmicas de fluxo de vento incidindo na lateral da carga
        if abs(Vv) > 0.2:
            len_w = np.clip(0.32 * abs(Vv), 1.8, 6.2)
            # Fase animada para dar sensação de escoamento de vento em movimento
            phase_shift = (tk * 3.5) % 1.5
            y_levels = np.linspace(cy_load - 0.42 * h_c, cy_load + 0.42 * h_c, n_wind_arrows)
            x_face_left = cx_load - 0.5 * p.L_c - 0.4
            for idx_w, w_line in enumerate(wind_arrows):
                x_tip = x_face_left - 0.3 * ((idx_w + phase_shift) % 1.5)
                x_tail = x_tip - len_w
                w_line.set_data([x_tail, x_tip], [y_levels[idx_w], y_levels[idx_w]])
            wind_label.set_position((x_face_left - len_w - 0.4, cy_load))
            wind_label.set_text(f"Vento $V_v = {Vv:.1f}$ m/s\n$F_{{vento}} = {F_wind:.0f}$ N")
        else:
            for w_line in wind_arrows:
                w_line.set_data([], [])
            wind_label.set_text("")

        # 6. Atualiza Telemetria (HUD) compacta no céu acima da lança
        hud_str = (
            f"TELEMETRIA EM TEMPO REAL  |  t = {tk:6.2f} s  |  V_vento = {Vv:5.2f} m/s\n"
            f"Trolleys: x1 = {x1:5.2f} m, x2 = {x2:5.2f} m  |  Cabos: T1 = {T1_kN:5.2f} kN, T2 = {T2_kN:5.2f} kN\n"
            f"Ângulos : theta1 = {th1_deg:+6.2f}°, theta2 = {th2_deg:+6.2f}°  |  Arfagem phi = {phi_deg:+6.3f}°"
        )
        hud_text.set_text(hud_str)

        # 7. Atualiza painel lateral sincronizado
        line_th1_dyn.set_data(t_all[: k + 1], th1_deg_all[: k + 1])
        line_th2_dyn.set_data(t_all[: k + 1], th2_deg_all[: k + 1])
        line_phi_dyn.set_data(t_all[: k + 1], phi_deg_all[: k + 1])
        cursor_theta.set_xdata([tk, tk])

        line_T1_dyn.set_data(t_all[: k + 1], T1_kN_all[: k + 1])
        line_T2_dyn.set_data(t_all[: k + 1], T2_kN_all[: k + 1])
        cursor_tens.set_xdata([tk, tk])

        return (
            trolley1_box,
            trolley2_box,
            *wheels,
            cable1_line,
            cable2_line,
            anchors_scatter,
            plat_poly,
            load_poly,
            trail_line,
            cg_marker,
            fm1_arrow,
            fm2_arrow,
            fm1_text,
            fm2_text,
            *wind_arrows,
            wind_label,
            hud_text,
            line_th1_dyn,
            line_th2_dyn,
            line_phi_dyn,
            cursor_theta,
            line_T1_dyn,
            line_T2_dyn,
            cursor_tens,
        )

    # Salva um quadro-chave representativo (no meio da manobra/rajada) como imagem de alta resolução para os slides
    mid_frame = int(0.35 * n_frames)
    update_frame(mid_frame)
    snapshot_path = FIGURES_DIR / f"{res.scenario_id}_frame_animacao.png"
    fig.savefig(snapshot_path, dpi=300, bbox_inches="tight")

    anim = mpl_anim.FuncAnimation(
        fig,
        update_frame,
        frames=n_frames,
        interval=int(1000.0 / fps),
        blit=False,
    )

    exported_paths: Dict[str, Path] = {"snapshot": snapshot_path}

    if save_mp4:
        mp4_path = output_dir / f"{res.scenario_id}_guindaste2d.mp4"
        writer_mp4 = mpl_anim.FFMpegWriter(fps=fps, bitrate=2400, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
        anim.save(str(mp4_path), writer=writer_mp4, dpi=130)
        exported_paths["mp4"] = mp4_path

    if save_gif:
        gif_path = output_dir / f"{res.scenario_id}_guindaste2d.gif"
        writer_gif = mpl_anim.PillowWriter(fps=min(fps, 20))
        anim.save(str(gif_path), writer=writer_gif, dpi=100)
        exported_paths["gif"] = gif_path

    if show_interactive:
        plt.show()
    else:
        plt.close(fig)

    return exported_paths

