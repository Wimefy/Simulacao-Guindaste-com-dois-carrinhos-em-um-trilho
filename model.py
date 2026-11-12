"""
model.py - Modelo Matemático Não-Linear em Espaço de Estados (Ordem 10).

Disciplina: PME3380 - Modelagem de Sistemas Dinâmicos (Poli-USP)
Projeto: Modelagem 2D de um Guindaste de Torre com Dois Trolleys no Mesmo Trilho
         segurando uma Plataforma com Carga.

Vetor de Estados (Ordem 10):
    x_n = [x1, dx1, x2, dx2, x_p, dx_p, y_p, dy_p, phi, dphi]^T

Entradas de Controle (Ordem 2):
    u_n = [F_M1(t), F_M2(t)]^T

Entrada de Perturbação (Ordem 1):
    w_n = [V_v(t)]^T

Vetor de Saídas (Ordem 7):
    y_n = [theta1, dtheta1, theta2, dtheta2, x_p, y_p, phi]^T
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from config import CraneParams, DEFAULT_PARAMS


STATE_NAMES = [
    "x1", "dx1", "x2", "dx2", "x_p", "dx_p", "y_p", "dy_p", "phi", "dphi"
]
STATE_UNITS = [
    "m", "m/s", "m", "m/s", "m", "m/s", "m", "m/s", "rad", "rad/s"
]
OUTPUT_NAMES = [
    "theta1", "dtheta1", "theta2", "dtheta2", "x_p", "y_p", "phi"
]
OUTPUT_UNITS = [
    "rad", "rad/s", "rad", "rad/s", "m", "m", "rad"
]


def get_equilibrium(
    params: CraneParams = DEFAULT_PARAMS,
    x1_eq: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula o ponto de equilíbrio estático exato (x_eq, u_eq, w_eq, y_eq).

    Parâmetros:
        params: Instância de `CraneParams`.
        x1_eq: Posição de equilíbrio do Trolley 1 [m] (padrão: `params.x1_eq_ref` = 10.0 m).

    Retorna:
        x_eq (10,), u_eq (2,), w_eq (1,), y_eq (7,)
    """
    if x1_eq is None:
        x1_eq = params.x1_eq_ref

    x2_eq = x1_eq + params.L_p
    x_p_eq = x1_eq + 0.5 * params.L_p
    y_p_eq = params.L_s + (params.m_p + params.m_c) * params.g / (2.0 * params.k)
    phi_eq = 0.0

    x_eq = np.array(
        [x1_eq, 0.0, x2_eq, 0.0, x_p_eq, 0.0, y_p_eq, 0.0, phi_eq, 0.0],
        dtype=float,
    )
    u_eq = np.array([0.0, 0.0], dtype=float)
    w_eq = np.array([0.0], dtype=float)
    y_eq = np.array([0.0, 0.0, 0.0, 0.0, x_p_eq, y_p_eq, phi_eq], dtype=float)
    return x_eq, u_eq, w_eq, y_eq


def compute_kinematics_and_forces(
    x: np.ndarray,
    u: np.ndarray,
    w: np.ndarray,
    params: CraneParams = DEFAULT_PARAMS,
) -> Dict[str, float]:
    """
    Calcula toda a cinemática dos pontos de ancoragem A e B, comprimentos e ângulos
    dos cabos (l1, l2, theta1, theta2), suas derivadas analíticas exatas
    (dl1, dl2, dtheta1, dtheta2), as trações viscoelásticas unilaterais (T1, T2)
    e as forças de atrito e aerodinâmica do vento.
    """
    x1, dx1, x2, dx2, x_p, dx_p, y_p, dy_p, phi, dphi = x
    F_M1, F_M2 = u
    V_v = float(w[0]) if np.ndim(w) > 0 else float(w)

    half_Lp = 0.5 * params.L_p
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # 1. Posições dos pontos de ancoragem A (esquerda) e B (direita) na plataforma
    x_A = x_p - half_Lp * cos_phi
    y_A = y_p - half_Lp * sin_phi
    x_B = x_p + half_Lp * cos_phi
    y_B = y_p + half_Lp * sin_phi

    # 2. Velocidades analíticas exatas dos pontos de ancoragem A e B
    dx_A = dx_p + half_Lp * sin_phi * dphi
    dy_A = dy_p - half_Lp * cos_phi * dphi
    dx_B = dx_p - half_Lp * sin_phi * dphi
    dy_B = dy_p + half_Lp * cos_phi * dphi

    # 3. Vetores relativos dos cabos (do trolley i em (x_i, 0) até o ponto de ancoragem)
    rx1 = x_A - x1
    ry1 = y_A
    drx1 = dx_A - dx1
    dry1 = dy_A

    rx2 = x_B - x2
    ry2 = y_B
    drx2 = dx_B - dx2
    dry2 = dy_B

    # 4. Comprimentos e ângulos dos cabos em relação ao eixo vertical Y (para baixo)
    l1 = float(np.hypot(rx1, ry1))
    l2 = float(np.hypot(rx2, ry2))
    theta1 = float(np.arctan2(rx1, ry1))
    theta2 = float(np.arctan2(rx2, ry2))

    # 5. Derivadas analíticas exatas de comprimento (dl_i) e de ângulo (dtheta_i)
    # l_i^2 = rx_i^2 + ry_i^2  =>  dl_i = (rx_i * drx_i + ry_i * dry_i) / l_i
    # theta_i = atan2(rx_i, ry_i) => dtheta_i = (drx_i * ry_i - rx_i * dry_i) / l_i^2
    dl1 = float((rx1 * drx1 + ry1 * dry1) / l1)
    dl2 = float((rx2 * drx2 + ry2 * dry2) / l2)

    dtheta1 = float((drx1 * ry1 - rx1 * dry1) / (l1 * l1))
    dtheta2 = float((drx2 * ry2 - rx2 * dry2) / (l2 * l2))

    # 6. Trações nos cabos (atuam apenas sob tração: T_i >= 0)
    T1_raw = params.k * (l1 - params.L_s) + params.c_s * dl1
    T2_raw = params.k * (l2 - params.L_s) + params.c_s * dl2
    T1 = float(max(0.0, T1_raw))
    T2 = float(max(0.0, T2_raw))

    # 7. Força aerodinâmica do vento sobre a lateral da carga (direção horizontal X)
    v_rel_vento = V_v - dx_p
    F_vento = float(
        0.5 * params.rho_ar * params.C_d * params.A_c * v_rel_vento * abs(v_rel_vento)
    )

    # 8. Forças normais e de atrito nos trolleys 1 e 2
    N1 = params.m_T * params.g + T1 * np.cos(theta1)
    N2 = params.m_T * params.g + T2 * np.cos(theta2)
    F_atrito1 = float(params.b_T * dx1 + params.mu_T * N1 * np.tanh(dx1 / 0.01))
    F_atrito2 = float(params.b_T * dx2 + params.mu_T * N2 * np.tanh(dx2 / 0.01))

    return {
        "x_A": float(x_A),
        "y_A": float(y_A),
        "x_B": float(x_B),
        "y_B": float(y_B),
        "dx_A": float(dx_A),
        "dy_A": float(dy_A),
        "dx_B": float(dx_B),
        "dy_B": float(dy_B),
        "l1": l1,
        "l2": l2,
        "dl1": dl1,
        "dl2": dl2,
        "theta1": theta1,
        "theta2": theta2,
        "dtheta1": dtheta1,
        "dtheta2": dtheta2,
        "T1": T1,
        "T2": T2,
        "F_vento": F_vento,
        "N1": float(N1),
        "N2": float(N2),
        "F_atrito1": F_atrito1,
        "F_atrito2": F_atrito2,
        "F_M1": float(F_M1),
        "F_M2": float(F_M2),
        "V_v": V_v,
    }


def f_nonlinear(
    t: float,
    x: np.ndarray,
    u: np.ndarray,
    w: np.ndarray,
    params: CraneParams = DEFAULT_PARAMS,
) -> np.ndarray:
    """
    Campo vetorial não-linear dx_n/dt = f(x_n, u_n, w_n) de ordem 10.

    Equações de Movimento:
      1. m_T * ddx1 = F_M1 - b_T*dx1 - mu_T*(m_T*g + T1*cos(theta1))*tanh(dx1/0.01) + T1*sin(theta1)
      2. m_T * ddx2 = F_M2 - b_T*dx2 - mu_T*(m_T*g + T2*cos(theta2))*tanh(dx2/0.01) + T2*sin(theta2)
      3. (m_p + m_c) * ddx_p = -T1*sin(theta1) - T2*sin(theta2) + F_vento
      4. (m_p + m_c) * ddy_p = (m_p + m_c)*g - T1*cos(theta1) - T2*cos(theta2)
      5. I_G * ddphi = (L_p/2)*cos(phi)*(T1*cos(theta1) - T2*cos(theta2))
                       + (L_p/2)*sin(phi)*(T2*sin(theta2) - T1*sin(theta1))
                       - F_vento*(h_c/2)*cos(phi)
    """
    _, dx1, _, dx2, _, dx_p, _, dy_p, phi, dphi = x
    F_M1, F_M2 = u

    kin = compute_kinematics_and_forces(x, u, w, params)
    T1 = kin["T1"]
    T2 = kin["T2"]
    theta1 = kin["theta1"]
    theta2 = kin["theta2"]
    F_vento = kin["F_vento"]

    cos_t1 = np.cos(theta1)
    sin_t1 = np.sin(theta1)
    cos_t2 = np.cos(theta2)
    sin_t2 = np.sin(theta2)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    M_total = params.m_p + params.m_c
    h_c = params.A_c / params.L_c
    half_Lp = 0.5 * params.L_p

    # Equação 1: Aceleração horizontal do Trolley 1
    ddx1 = (
        F_M1
        - params.b_T * dx1
        - params.mu_T * (params.m_T * params.g + T1 * cos_t1) * np.tanh(dx1 / 0.01)
        + T1 * sin_t1
    ) / params.m_T

    # Equação 2: Aceleração horizontal do Trolley 2
    ddx2 = (
        F_M2
        - params.b_T * dx2
        - params.mu_T * (params.m_T * params.g + T2 * cos_t2) * np.tanh(dx2 / 0.01)
        + T2 * sin_t2
    ) / params.m_T

    # Equação 3: Aceleração horizontal da plataforma + carga
    ddx_p = (-T1 * sin_t1 - T2 * sin_t2 + F_vento) / M_total

    # Equação 4: Aceleração vertical da plataforma + carga (eixo Y aponta para baixo)
    ddy_p = params.g - (T1 * cos_t1 + T2 * cos_t2) / M_total

    # Equação 5: Aceleração angular (arfagem phi) da plataforma + carga
    ddphi = (
        half_Lp * cos_phi * (T1 * cos_t1 - T2 * cos_t2)
        + half_Lp * sin_phi * (T2 * sin_t2 - T1 * sin_t1)
        - F_vento * (0.5 * h_c) * cos_phi
    ) / params.I_G

    return np.array(
        [dx1, ddx1, dx2, ddx2, dx_p, ddx_p, dy_p, ddy_p, dphi, ddphi],
        dtype=float,
    )


def g_output(
    x: np.ndarray,
    u: np.ndarray | None = None,
    w: np.ndarray | None = None,
    params: CraneParams = DEFAULT_PARAMS,
) -> np.ndarray:
    """
    Vetor de saídas não-linear y_n = [theta1, dtheta1, theta2, dtheta2, x_p, y_p, phi]^T.
    """
    if u is None:
        u = np.zeros(2, dtype=float)
    if w is None:
        w = np.zeros(1, dtype=float)

    kin = compute_kinematics_and_forces(x, u, w, params)
    return np.array(
        [
            kin["theta1"],
            kin["dtheta1"],
            kin["theta2"],
            kin["dtheta2"],
            float(x[4]),  # x_p
            float(x[6]),  # y_p
            float(x[8]),  # phi
        ],
        dtype=float,
    )


def compute_energy(
    x: np.ndarray,
    params: CraneParams = DEFAULT_PARAMS,
) -> Dict[str, float]:
    """
    Calcula as energias cinética, potencial gravitacional, potencial elástica
    (referenciadas ao ponto de equilíbrio estático) e a potência instantânea dissipada.

    Nota: Como o eixo Y aponta para baixo (no sentido de g), a energia potencial
    gravitacional relativa ao equilíbrio y_p_eq é:
        V_grav = -M * g * (y_p - y_p_eq)
    E a energia elástica acumulada nos cabos (comportamento unilateral l_i > L_s) é:
        V_elast = 0.5 * k * max(0, l1 - L_s)^2 + 0.5 * k * max(0, l2 - L_s)^2 - V_elast_eq
    No equilíbrio estático (l1 = l2 = y_p_eq), V_pot = V_grav + V_elast tem mínimo global zero!
    """
    _, dx1, _, dx2, _, dx_p, y_p, dy_p, _, dphi = x
    kin = compute_kinematics_and_forces(x, np.zeros(2), np.zeros(1), params)

    M_total = params.m_p + params.m_c
    y_p_eq = params.y_p_eq
    dl_eq = y_p_eq - params.L_s

    # Energia cinética total: Trolleys + Translação da Plataforma/Carga + Rotação
    K_trolleys = 0.5 * params.m_T * (dx1 ** 2 + dx2 ** 2)
    K_trans = 0.5 * M_total * (dx_p ** 2 + dy_p ** 2)
    K_rot = 0.5 * params.I_G * (dphi ** 2)
    E_cin = float(K_trolleys + K_trans + K_rot)

    # Energia potencial gravitacional + elástica (relativa ao equilíbrio estático)
    ext1 = max(0.0, kin["l1"] - params.L_s)
    ext2 = max(0.0, kin["l2"] - params.L_s)
    V_elast_eq = 2.0 * (0.5 * params.k * (dl_eq ** 2))
    V_elast = 0.5 * params.k * (ext1 ** 2 + ext2 ** 2) - V_elast_eq
    V_grav = -M_total * params.g * (y_p - y_p_eq)
    E_pot = float(V_elast + V_grav)

    # Potência dissipada nos cabos e nos trolleys [W]
    P_dissip_cabos = params.c_s * (
        (kin["dl1"] ** 2 if kin["T1"] > 0 else 0.0)
        + (kin["dl2"] ** 2 if kin["T2"] > 0 else 0.0)
    )
    P_dissip_trolleys = kin["F_atrito1"] * dx1 + kin["F_atrito2"] * dx2

    return {
        "E_cin": E_cin,
        "E_pot": E_pot,
        "E_total": E_cin + E_pot,
        "K_trolleys": float(K_trolleys),
        "K_trans": float(K_trans),
        "K_rot": float(K_rot),
        "P_dissip": float(P_dissip_cabos + P_dissip_trolleys),
    }

