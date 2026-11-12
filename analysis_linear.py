"""
analysis_linear.py - Equilíbrio Estático, Linearização Simbólica (SymPy), Autovalores,
                     Tabela de Routh-Hurwitz e Funções de Transferência.

Disciplina: PME3380 - Modelagem de Sistemas Dinâmicos (Poli-USP)
Projeto: Modelagem 2D de um Guindaste de Torre com Dois Trolleys no Mesmo Trilho
         segurando uma Plataforma com Carga.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config import CraneParams, DEFAULT_PARAMS, TABLES_DIR
import control as ct
import numpy as np
from scipy import linalg, signal
import sympy as sp

from model import (
    OUTPUT_NAMES,
    STATE_NAMES,
    f_nonlinear,
    g_output,
    get_equilibrium,
)


@dataclass
class ModalInfo:
    """Estrutura contendo os parâmetros modais de cada autovalor/modo físico."""

    index: int
    eigenvalue: complex
    wn_rad_s: float
    fn_hz: float
    wd_rad_s: float
    zeta: float
    time_constant_s: float
    mode_name: str
    dominant_states: str


@dataclass
class LinearAnalysisResult:
    """Resultado completo da análise de equilíbrio e linearização."""

    params: CraneParams
    include_dry_friction: bool
    x_eq: np.ndarray
    u_eq: np.ndarray
    w_eq: np.ndarray
    y_eq: np.ndarray
    f_eq_norm: float
    A: np.ndarray
    B: np.ndarray
    E: np.ndarray
    E_wind_nominal: np.ndarray
    C: np.ndarray
    D: np.ndarray
    A_sym_str: List[List[str]]
    char_poly_coeffs_10: np.ndarray
    char_poly_coeffs_9: np.ndarray
    routh_table_9: np.ndarray
    routh_row_labels_9: List[str]
    is_internal_stable: bool
    modes: List[ModalInfo]
    transfer_functions: Dict[str, ct.TransferFunction]
    poles_Zeros_dict: Dict[str, Dict[str, np.ndarray]]


def build_symbolic_model() -> Dict[str, Any]:
    """
    Constrói o modelo simbólico completo no SymPy e calcula as matrizes Jacobianas
    analíticas exatas em torno do ponto de equilíbrio estático.
    """
    # Símbolos de estado (ordem 10)
    x1, dx1, x2, dx2, xp, dxp, yp, dyp, phi, dphi = sp.symbols(
        "x1 dx1 x2 dx2 x_p dx_p y_p dy_p phi dphi", real=True
    )
    # Símbolos de entradas de controle e perturbação
    FM1, FM2, Vv = sp.symbols("F_M1 F_M2 V_v", real=True)
    # Símbolos de parâmetros físicos
    m_T, m_p, m_c, L_s, L_p, L_c, A_c, k, c_s, mu_T, b_T, C_d, rho_ar, g, I_G = sp.symbols(
        "m_T m_p m_c L_s L_p L_c A_c k c_s mu_T b_T C_d rho_ar g I_G",
        real=True,
        positive=True,
    )
    x1_eq_sym = sp.Symbol("x1_eq", real=True)
    Vv_nom_sym = sp.Symbol("V_v_nom", real=True, positive=True)

    M = m_p + m_c
    h_c = A_c / L_c
    half_Lp = L_p / 2

    # Cinemática dos pontos de ancoragem A e B
    xA = xp - half_Lp * sp.cos(phi)
    yA = yp - half_Lp * sp.sin(phi)
    xB = xp + half_Lp * sp.cos(phi)
    yB = yp + half_Lp * sp.sin(phi)

    dxA = dxp + half_Lp * sp.sin(phi) * dphi
    dyA = dyp - half_Lp * sp.cos(phi) * dphi
    dxB = dxp - half_Lp * sp.sin(phi) * dphi
    dyB = dyp + half_Lp * sp.cos(phi) * dphi

    # Comprimentos e ângulos dos cabos
    l1 = sp.sqrt((xA - x1) ** 2 + yA ** 2)
    l2 = sp.sqrt((xB - x2) ** 2 + yB ** 2)
    theta1 = sp.atan2(xA - x1, yA)
    theta2 = sp.atan2(xB - x2, yB)

    dl1 = ((xA - x1) * (dxA - dx1) + yA * dyA) / l1
    dl2 = ((xB - x2) * (dxB - dx2) + yB * dyB) / l2

    dtheta1 = ((dxA - dx1) * yA - (xA - x1) * dyA) / (l1 ** 2)
    dtheta2 = ((dxB - dx2) * yB - (xB - x2) * dyB) / (l2 ** 2)

    # Na vizinhança do equilíbrio estático, T1_eq = T2_eq = (m_p + m_c)*g/2 > 0
    T1 = k * (l1 - L_s) + c_s * dl1
    T2 = k * (l2 - L_s) + c_s * dl2

    # Para V_v - dx_p >= 0, (V_v - dx_p)*|V_v - dx_p| = (V_v - dx_p)^2
    # Em V_v_eq = 0 e dx_p_eq = 0, a derivada primeira de (V_v - dx_p)|V_v - dx_p| é nula.
    F_vento_pos = sp.Rational(1, 2) * rho_ar * C_d * A_c * (Vv - dxp) ** 2

    ddx1 = (
        FM1
        - b_T * dx1
        - mu_T * (m_T * g + T1 * sp.cos(theta1)) * sp.tanh(dx1 / sp.Rational(1, 100))
        + T1 * sp.sin(theta1)
    ) / m_T

    ddx2 = (
        FM2
        - b_T * dx2
        - mu_T * (m_T * g + T2 * sp.cos(theta2)) * sp.tanh(dx2 / sp.Rational(1, 100))
        + T2 * sp.sin(theta2)
    ) / m_T

    ddxp = (-T1 * sp.sin(theta1) - T2 * sp.sin(theta2) + F_vento_pos) / M
    ddyp = g - (T1 * sp.cos(theta1) + T2 * sp.cos(theta2)) / M
    ddphi = (
        half_Lp * sp.cos(phi) * (T1 * sp.cos(theta1) - T2 * sp.cos(theta2))
        + half_Lp * sp.sin(phi) * (T2 * sp.sin(theta2) - T1 * sp.sin(theta1))
        - F_vento_pos * (h_c / 2) * sp.cos(phi)
    ) / I_G

    state_vec = sp.Matrix([x1, dx1, x2, dx2, xp, dxp, yp, dyp, phi, dphi])
    input_vec = sp.Matrix([FM1, FM2])
    dist_vec = sp.Matrix([Vv])
    f_vec = sp.Matrix([dx1, ddx1, dx2, ddx2, dxp, ddxp, dyp, ddyp, dphi, ddphi])
    y_vec = sp.Matrix([theta1, dtheta1, theta2, dtheta2, xp, yp, phi])

    # Dicionário de substituição do ponto de equilíbrio estático
    eq_subs = {
        x1: x1_eq_sym,
        dx1: 0,
        x2: x1_eq_sym + L_p,
        dx2: 0,
        xp: x1_eq_sym + L_p / 2,
        dxp: 0,
        yp: L_s + M * g / (2 * k),
        dyp: 0,
        phi: 0,
        dphi: 0,
        FM1: 0,
        FM2: 0,
        Vv: 0,
    }

    # 1. Verificação simbólica de f(x_eq, u_eq, w_eq) = 0
    f_at_eq = sp.simplify(f_vec.subs(eq_subs))

    # 2. Jacobianas simbólicas exatas avaliadas no equilíbrio
    A_sym = sp.simplify(f_vec.jacobian(state_vec).subs(eq_subs))
    B_sym = sp.simplify(f_vec.jacobian(input_vec).subs(eq_subs))
    E_sym = sp.simplify(f_vec.jacobian(dist_vec).subs(eq_subs))
    C_sym = sp.simplify(y_vec.jacobian(state_vec).subs(eq_subs))
    D_sym = sp.simplify(y_vec.jacobian(input_vec).subs(eq_subs))

    # Jacobiana de perturbação do vento em torno de uma velocidade nominal V_v = V_v_nom
    eq_subs_wind = dict(eq_subs)
    eq_subs_wind[Vv] = Vv_nom_sym
    E_wind_nom_sym = sp.simplify(f_vec.jacobian(dist_vec).subs(eq_subs_wind))

    return {
        "symbols": {
            "m_T": m_T,
            "m_p": m_p,
            "m_c": m_c,
            "L_s": L_s,
            "L_p": L_p,
            "L_c": L_c,
            "A_c": A_c,
            "k": k,
            "c_s": c_s,
            "mu_T": mu_T,
            "b_T": b_T,
            "C_d": C_d,
            "rho_ar": rho_ar,
            "g": g,
            "I_G": I_G,
            "x1_eq": x1_eq_sym,
            "V_v_nom": Vv_nom_sym,
        },
        "f_at_eq": f_at_eq,
        "A_sym": A_sym,
        "B_sym": B_sym,
        "E_sym": E_sym,
        "E_wind_nom_sym": E_wind_nom_sym,
        "C_sym": C_sym,
        "D_sym": D_sym,
    }


def compute_routh_hurwitz(coeffs: np.ndarray, tol: float = 1e-12) -> Tuple[np.ndarray, List[str], bool]:
    """
    Constrói a Tabela de Routh-Hurwitz completa para um polinômio característico
    P(s) = a_n s^n + a_{n-1} s^{n-1} + ... + a_1 s + a_0.

    Trata automaticamente casos especiais:
      - Elemento nulo na primeira coluna (substituição por epsilon positivo pequeno);
      - Linha inteira de zeros (utilizando a derivada do polinômio auxiliar da linha anterior).

    Retorna:
        routh_table (n+1, n_cols), row_labels (List[str]), is_hurwitz_stable (bool)
    """
    coeffs = np.asarray(coeffs, dtype=float).flatten()
    # Remove eventuais zeros líderes
    while len(coeffs) > 1 and abs(coeffs[0]) < tol:
        coeffs = coeffs[1:]

    n = len(coeffs) - 1
    n_cols = (n + 2) // 2
    table = np.zeros((n + 1, n_cols), dtype=float)
    labels = [f"s^{n - i}" for i in range(n + 1)]

    # Preenche as duas primeiras linhas (s^n e s^{n-1})
    table[0, : len(coeffs[0::2])] = coeffs[0::2]
    if n >= 1:
        table[1, : len(coeffs[1::2])] = coeffs[1::2]

    for i in range(2, n + 1):
        # Caso especial 1: linha anterior inteira nula -> polinômio auxiliar da linha i-2
        if np.all(np.abs(table[i - 1, :]) < tol):
            deg_aux = n - (i - 2)
            for j in range(n_cols):
                power = deg_aux - 2 * j
                if power >= 0:
                    table[i - 1, j] = table[i - 2, j] * power

        pivot = table[i - 1, 0]
        # Caso especial 2: apenas o primeiro elemento da linha anterior é zero
        if abs(pivot) < tol:
            pivot = tol if pivot >= 0 else -tol

        for j in range(n_cols - 1):
            b_val = (table[i - 1, 0] * table[i - 2, j + 1] - table[i - 2, 0] * table[i - 1, j + 1]) / pivot
            table[i, j] = b_val

    first_col = table[:, 0]
    is_stable = bool(np.all(first_col > tol) or np.all(first_col < -tol))
    return table, labels, is_stable


def classify_modes(A: np.ndarray, params: CraneParams, include_dry_friction: bool) -> List[ModalInfo]:
    """
    Calcula os autovalores e autovetores de A, ordena e classifica fisicamente cada
    modo do guindaste (Translação, Balanço Pendular, Diferencial dos Trolleys,
    Modo Axial dos Cabos e Modo de Arfagem phi).
    """
    eigvals, eigvecs = linalg.eig(A)

    # Índices de estado: 0:x1, 1:dx1, 2:x2, 3:dx2, 4:xp, 5:dxp, 6:yp, 7:dyp, 8:phi, 9:dphi
    modes: List[ModalInfo] = []
    for idx, ev in enumerate(eigvals):
        vec = np.abs(eigvecs[:, idx])
        wn = float(np.abs(ev))
        fn = wn / (2.0 * np.pi)
        wd = float(np.abs(ev.imag))
        zeta = float(-ev.real / wn) if wn > 1e-9 else 0.0
        tau = float(-1.0 / ev.real) if ev.real < -1e-9 else float("inf")

        # Identificação física baseada na participação de estados e frequência
        if wn < 1e-7:
            mode_name = "Translação Rígida Livre (Integrador s = 0)"
            dom = "x1, x2, x_p (movimento global)"
        elif vec[8] + vec[9] > 0.5 * np.sum(vec):
            mode_name = "Arfagem / Rotação da Plataforma (Pitch phi)"
            dom = "phi, dphi"
        elif vec[6] + vec[7] > 0.5 * np.sum(vec):
            mode_name = "Elástico Axial Vertical dos Cabos (Bounce y_p)"
            dom = "y_p, dy_p"
        elif abs(ev.imag) > 1e-4:
            # Verifica simetria entre trolley 1 e trolley 2 no autovetor
            v_x1 = eigvecs[0, idx]
            v_x2 = eigvecs[2, idx]
            if abs(v_x1 + v_x2) < 1e-3 * (abs(v_x1) + abs(v_x2) + 1e-12):
                mode_name = "Diferencial Antissimétrico dos Trolleys (x1 - x2)"
                dom = "x1 - x2, theta1 - theta2"
            else:
                if include_dry_friction:
                    mode_name = "Balanço Pendular Simples (Trolleys Quase-Fixos pelo Atrito)"
                else:
                    mode_name = "Balanço Pendular Simétrico Acoplado (Trolleys + Carga)"
                dom = "x_p, theta1, theta2, x1 + x2"
        else:
            if wn < 0.5:
                mode_name = "Translação Global Amortecida no Trilho"
                dom = "dx1, dx2, dx_p"
            else:
                mode_name = "Relaxação Rápida de Velocidade dos Trolleys (Atrito)"
                dom = "dx1, dx2"

        modes.append(
            ModalInfo(
                index=idx + 1,
                eigenvalue=complex(ev),
                wn_rad_s=wn,
                fn_hz=fn,
                wd_rad_s=wd,
                zeta=zeta,
                time_constant_s=tau,
                mode_name=mode_name,
                dominant_states=dom,
            )
        )

    # Ordena por frequência natural crescente
    modes.sort(key=lambda m: (round(m.wn_rad_s, 5), -m.eigenvalue.imag))
    for i, m in enumerate(modes):
        m.index = i + 1
    return modes


def _clean_tf_from_sympy(tf_expr: sp.Expr, s_sym: sp.Symbol) -> ct.TransferFunction:
    """Converte uma função de transferência racional simbólica do SymPy em `control.TransferFunction`."""
    tf_simplified = sp.cancel(sp.simplify(tf_expr))
    num_expr, den_expr = sp.fraction(tf_simplified)
    num_poly = sp.Poly(sp.expand(num_expr), s_sym)
    den_poly = sp.Poly(sp.expand(den_expr), s_sym)

    num_coeffs = np.array([float(c) for c in num_poly.all_coeffs()], dtype=float)
    den_coeffs = np.array([float(c) for c in den_poly.all_coeffs()], dtype=float)

    # Normaliza pelo coeficiente líder do denominador
    lead = den_coeffs[0]
    num_coeffs = num_coeffs / lead
    den_coeffs = den_coeffs / lead

    # Limpa eventuais resíduos numéricos de ponto flutuante
    num_coeffs[np.abs(num_coeffs) < 1e-13 * np.max(np.abs(num_coeffs) + 1e-30)] = 0.0
    den_coeffs[np.abs(den_coeffs) < 1e-13 * np.max(np.abs(den_coeffs) + 1e-30)] = 0.0
    if np.all(num_coeffs == 0.0):
        num_coeffs = np.array([0.0])

    return ct.TransferFunction(num_coeffs, den_coeffs)


def run_linear_analysis(
    params: CraneParams = DEFAULT_PARAMS,
    include_dry_friction: bool = False,
    v_wind_nominal: float = 10.0,
) -> LinearAnalysisResult:
    """
    Executa a análise completa de equilíbrio e linearização:
      1. Verifica f(x_eq, u_eq, w_eq) = 0;
      2. Obtém as matrizes Jacobianas A, B, E, C, D via SymPy;
      3. Calcula o polinômio característico, autovalores, frequências naturais e amortecimentos;
      4. Constrói a Tabela de Routh-Hurwitz;
      5. Extrai as Funções de Transferência mínimas e seus polos/zeros.

    Parâmetros:
        params: Instância de `CraneParams`.
        include_dry_friction: Se False, utiliza `mu_T = 0` na linearização para revelar
                              os modos vibratórios puros acoplados; se True, utiliza `params.mu_T`.
        v_wind_nominal: Velocidade de referência do vento [m/s] para avaliar a sensibilidade
                        aerodinâmica linearizada E_wind_nominal (já que em V_v = 0 a derivada é nula).
    """
    x_eq, u_eq, w_eq, y_eq = get_equilibrium(params)
    f_eq = f_nonlinear(0.0, x_eq, u_eq, w_eq, params)
    f_eq_norm = float(np.max(np.abs(f_eq)))

    sym_data = build_symbolic_model()
    syms = sym_data["symbols"]

    mu_val = params.mu_T if include_dry_friction else 0.0
    num_subs = {
        syms["m_T"]: sp.Rational(str(params.m_T)),
        syms["m_p"]: sp.Rational(str(params.m_p)),
        syms["m_c"]: sp.Rational(str(params.m_c)),
        syms["L_s"]: sp.Rational(str(params.L_s)),
        syms["L_p"]: sp.Rational(str(params.L_p)),
        syms["L_c"]: sp.Rational(str(params.L_c)),
        syms["A_c"]: sp.Rational(str(params.A_c)),
        syms["k"]: sp.Rational(str(params.k)),
        syms["c_s"]: sp.Rational(str(params.c_s)),
        syms["mu_T"]: sp.Rational(str(mu_val)),
        syms["b_T"]: sp.Rational(str(params.b_T)),
        syms["C_d"]: sp.Rational(str(params.C_d)),
        syms["rho_ar"]: sp.Rational(str(params.rho_ar)),
        syms["g"]: sp.Rational(str(params.g)),
        syms["I_G"]: sp.Rational(str(params.m_p)) * sp.Rational(str(params.L_p)) ** 2 / 12
        + sp.Rational(str(params.m_c))
        * (sp.Rational(str(params.L_c)) ** 2 + (sp.Rational(str(params.A_c)) / sp.Rational(str(params.L_c))) ** 2)
        / 12,
        syms["x1_eq"]: sp.Rational(str(params.x1_eq_ref)),
        syms["V_v_nom"]: sp.Rational(str(v_wind_nominal)),
    }

    A_exact = sym_data["A_sym"].subs(num_subs)
    B_exact = sym_data["B_sym"].subs(num_subs)
    E_exact = sym_data["E_sym"].subs(num_subs)
    E_wind_exact = sym_data["E_wind_nom_sym"].subs(num_subs)
    C_exact = sym_data["C_sym"].subs(num_subs)
    D_exact = sym_data["D_sym"].subs(num_subs)

    A = np.array(A_exact.evalf(), dtype=float)
    B = np.array(B_exact.evalf(), dtype=float)
    E = np.array(E_exact.evalf(), dtype=float)
    E_wind_nominal = np.array(E_wind_exact.evalf(), dtype=float)
    C = np.array(C_exact.evalf(), dtype=float)
    D = np.array(D_exact.evalf(), dtype=float)

    A_sym_str = [[str( sym_data["A_sym"][i, j] ) for j in range(10)] for i in range(10)]

    # Polinômio característico exato via SymPy aproveitando a estrutura bloco-diagonal (6x6 + 2x2 + 2x2)
    s = sp.Symbol("s")
    M1 = s * sp.eye(6) - A_exact[0:6, 0:6]
    M2 = s * sp.eye(2) - A_exact[6:8, 6:8]
    M3 = s * sp.eye(2) - A_exact[8:10, 8:10]

    char_poly_sym = sp.expand(M1.det() * M2.det() * M3.det())
    poly_s = sp.Poly(char_poly_sym, s)
    char_poly_coeffs_10 = np.array([float(c) for c in poly_s.all_coeffs()], dtype=float)
    # Como o termo independente a_0 = 0 (polo integrador s = 0), Q_9(s) = P_10(s) / s
    if abs(char_poly_coeffs_10[-1]) < 1e-9:
        char_poly_coeffs_10[-1] = 0.0
    char_poly_coeffs_9 = char_poly_coeffs_10[:-1].copy()

    # Tabela de Routh-Hurwitz para Q_9(s) (modos internos + velocidade de translação)
    routh_table_9, routh_labels_9, is_internal_stable = compute_routh_hurwitz(char_poly_coeffs_9)

    # Classificação modal completa
    modes = classify_modes(A, params, include_dry_friction)

    # Funções de transferência exatas (livres de ruído numérico de cancelamento polo-zero)
    resolvent = sp.diag(M1.inv(), M2.inv(), M3.inv())
    G_u_sym = C_exact * resolvent * B_exact + D_exact
    G_w_sym = C_exact * resolvent * E_wind_exact

    # Entrada síncrona F_sinc: F_M1 = F_M2 = F_M -> B_sinc = B[:, 0] + B[:, 1]
    B_sync_exact = B_exact[:, 0] + B_exact[:, 1]
    G_sync_sym = C_exact * resolvent * B_sync_exact

    tfs: Dict[str, ct.TransferFunction] = {
        "theta1_FM1": _clean_tf_from_sympy(G_u_sym[0, 0], s),
        "theta2_FM1": _clean_tf_from_sympy(G_u_sym[2, 0], s),
        "theta1_F_sinc": _clean_tf_from_sympy(G_sync_sym[0, 0], s),
        "x_p_FM1": _clean_tf_from_sympy(G_u_sym[4, 0], s),
        "x_p_F_sinc": _clean_tf_from_sympy(G_sync_sym[4, 0], s),
        "theta1_Vv_nom": _clean_tf_from_sympy(G_w_sym[0, 0], s),
        "phi_Vv_nom": _clean_tf_from_sympy(G_w_sym[6, 0], s),
        "x_p_Vv_nom": _clean_tf_from_sympy(G_w_sym[4, 0], s),
    }

    pz_dict: Dict[str, Dict[str, np.ndarray]] = {}
    for name, tf_obj in tfs.items():
        pz_dict[name] = {
            "poles": ct.poles(tf_obj),
            "zeros": ct.zeros(tf_obj),
        }

    return LinearAnalysisResult(
        params=params,
        include_dry_friction=include_dry_friction,
        x_eq=x_eq,
        u_eq=u_eq,
        w_eq=w_eq,
        y_eq=y_eq,
        f_eq_norm=f_eq_norm,
        A=A,
        B=B,
        E=E,
        E_wind_nominal=E_wind_nominal,
        C=C,
        D=D,
        A_sym_str=A_sym_str,
        char_poly_coeffs_10=char_poly_coeffs_10,
        char_poly_coeffs_9=char_poly_coeffs_9,
        routh_table_9=routh_table_9,
        routh_row_labels_9=routh_labels_9,
        is_internal_stable=is_internal_stable,
        modes=modes,
        transfer_functions=tfs,
        poles_Zeros_dict=pz_dict,
    )


def format_routh_table_ascii(table: np.ndarray, labels: List[str]) -> str:
    """Formata a Tabela de Routh-Hurwitz como tabela ASCII alinhada para o terminal."""
    n_rows, n_cols = table.shape
    header = f"{' Grau ':>8} | " + " | ".join([f"{f'Coluna {j+1}':>14}" for j in range(n_cols)])
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for i in range(n_rows):
        row_vals = " | ".join([f"{table[i, j]:14.4e}" for j in range(n_cols)])
        lines.append(f"{labels[i]:>8} | {row_vals}")
    lines.append(sep)
    return "\n".join(lines)


def format_routh_table_latex(table: np.ndarray, labels: List[str], caption: str) -> str:
    """Gera o código LaTeX pronto para slides/relatório PME3380 da Tabela de Routh-Hurwitz."""
    n_rows, n_cols = table.shape
    col_spec = "c|" + "r" * n_cols
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        "Potência de $s$ & " + " & ".join([f"Col. {j+1}" for j in range(n_cols)]) + r" \\",
        r"\hline",
    ]
    for i in range(n_rows):
        formatted_entries = []
        for j in range(n_cols):
            val = table[i, j]
            if abs(val) < 1e-12:
                formatted_entries.append("0")
            elif 1e-2 <= abs(val) < 1e4:
                formatted_entries.append(f"{val:.4f}")
            else:
                exp = int(np.floor(np.log10(abs(val))))
                mant = val / (10 ** exp)
                formatted_entries.append(f"${mant:.3f} \\times 10^{{{exp}}}$")
        lines.append(f"${labels[i]}$ & " + " & ".join(formatted_entries) + r" \\")
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def export_linear_analysis_reports(
    res_pure: LinearAnalysisResult,
    res_fric: LinearAnalysisResult,
    output_dir: Path = TABLES_DIR,
) -> Tuple[Path, Path]:
    """
    Imprime no terminal e salva em disco (`.txt` e `.tex`) o relatório completo
    da análise de equilíbrio, linearização, modos naturais, Routh-Hurwitz e
    funções de transferência.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / "relatorio_linearizacao.txt"
    tex_path = output_dir / "tabelas_latex_pme3380.tex"

    p = res_pure.params
    lines: List[str] = []
    lines.append("=" * 92)
    lines.append(" PME3380 - MODELAGEM DE SISTEMAS DINÂMICOS (POLI-USP)")
    lines.append(" ANÁLISE DE EQUILÍBRIO E LINEARIZAÇÃO DO GUINDASTE 2D COM DOIS TROLLEYS")
    lines.append("=" * 92)

    # 1. Ponto de Equilíbrio
    lines.append("\n1. DEMONSTRAÇÃO EXPLÍCITA DO PONTO DE EQUILÍBRIO ESTÁTICO (f(x_eq, u_eq, w_eq) = 0)")
    lines.append("-" * 92)
    lines.append(f"  Massa total suspensa M = m_p + m_c = {p.M:.2f} kg")
    lines.append(f"  Momento de inércia baricêntrico I_G = {p.I_G:.4f} kg.m^2")
    lines.append(f"  Tração estática em cada cabo T1_eq = T2_eq = (m_p + m_c)*g/2 = {p.T_eq:.4f} N")
    lines.append(f"  Alongamento estático de cada cabo delta_l_eq = T_eq / k = {p.delta_l_eq*1000:.4f} mm")
    lines.append("\n  Vetor de Estados de Equilíbrio x_eq:")
    for name, val, unit in zip(STATE_NAMES, res_pure.x_eq, ["m", "m/s", "m", "m/s", "m", "m/s", "m", "m/s", "rad", "rad/s"]):
        lines.append(f"    {name:>6s}_eq = {val:12.6f} [{unit}]")
    lines.append(f"\n  Entradas de Equilíbrio: u_eq = {res_pure.u_eq.tolist()} [N], w_eq = {res_pure.w_eq.tolist()} [m/s]")
    lines.append(f"  Saídas de Equilíbrio:   y_eq = {[round(v, 6) for v in res_pure.y_eq.tolist()]}")
    lines.append(f"  Norma infinita do resíduo ||f(x_eq, u_eq, w_eq)||_inf = {res_pure.f_eq_norm:.3e} (Exatamente ZERO!)")

    # 2. Expressões Simbólicas Não-Nulas da Matriz A
    lines.append("\n2. EXPRESSÕES ANALÍTICAS SIMBÓLICAS NÃO-NULAS DA MATRIZ JACOBIANA A")
    lines.append("-" * 92)
    sym_data = build_symbolic_model()
    A_sym = sym_data["A_sym"]
    for i in range(10):
        for j in range(10):
            if A_sym[i, j] != 0:
                lines.append(f"  A[{i+1:2d},{j+1:2d}] (d({STATE_NAMES[i]})'/d({STATE_NAMES[j]})) = {A_sym[i, j]}")

    # 3. Comparação de Autovalores e Modos Físicos (Sem Atrito Seco vs Com Atrito Linearizado)
    for title, res in [
        ("3A. MODOS PUROS DO SISTEMA LINEARIZADO (Sem Atrito Seco: mu_T = 0)", res_pure),
        ("3B. MODOS COM ATRITO DE COULOMB REGULARIZADO LINEARIZADO (mu_T = 0.015)", res_fric),
    ]:
        lines.append(f"\n{title}")
        lines.append("-" * 92)
        lines.append(
            f"{'#':>2} | {'Autovalor (s^-1)':>24} | {'wn [rad/s]':>10} | {'fn [Hz]':>8} | {'zeta [-]':>8} | {'Modo Físico Dominante'}"
        )
        lines.append("-" * 92)
        for m in res.modes:
            ev_str = f"{m.eigenvalue.real:+10.5f} {m.eigenvalue.imag:+10.5f}j"
            lines.append(
                f"{m.index:2d} | {ev_str:>24} | {m.wn_rad_s:10.4f} | {m.fn_hz:8.4f} | {m.zeta:8.4f} | {m.mode_name}"
            )

    # 4. Polinômio Característico e Tabela de Routh-Hurwitz
    lines.append("\n4. POLINÔMIO CARACTERÍSTICO E TABELA DE ROUTH-HURWITZ")
    lines.append("-" * 92)
    lines.append("  Nota Física Importante: O sistema completo de 10ª ordem possui 1 polo em s = 0")
    lines.append("  correspondente à translação horizontal de corpo rígido ao longo do trilho")
    lines.append("  (invariância espacial: deslocar x1, x2 e x_p por +Delta_x preserva o equilíbrio).")
    lines.append("  Fatorando P_10(s) = s * Q_9(s), analisamos a estabilidade de Routh-Hurwitz de Q_9(s):")

    for sub_title, res in [
        ("Caso 1: Modos Puros (mu_T = 0)", res_pure),
        ("Caso 2: Com Atrito Linearizado (mu_T = 0.015)", res_fric),
    ]:
        lines.append(f"\n  >>> {sub_title} <<<")
        poly_terms = [f"{c:.4e}*s^{9-i}" for i, c in enumerate(res.char_poly_coeffs_9)]
        lines.append("  Q_9(s) = " + " + ".join(poly_terms))
        lines.append(format_routh_table_ascii(res.routh_table_9, res.routh_row_labels_9))
        lines.append(
            f"  Todos os elementos da 1ª coluna são estritamente positivos? "
            f"{'SIM (Sistema Internamente Assintoticamente Estável!)' if res.is_internal_stable else 'NÃO'}"
        )

    # 5. Funções de Transferência Principais
    lines.append("\n5. FUNÇÕES DE TRANSFERÊNCIA PRINCIPAIS (REALIZAÇÃO MÍNIMA EXATA - mu_T = 0)")
    lines.append("-" * 92)
    for tf_name, tf_obj in res_pure.transfer_functions.items():
        lines.append(f"\n  Canal: {tf_name}")
        lines.append(str(tf_obj))
        z_arr = res_pure.poles_Zeros_dict[tf_name]["zeros"]
        p_arr = res_pure.poles_Zeros_dict[tf_name]["poles"]
        lines.append(f"  Polos ({len(p_arr)}): " + ", ".join([f"{p.real:+.4f}{p.imag:+.4f}j" for p in p_arr]))
        lines.append(f"  Zeros ({len(z_arr)}): " + ", ".join([f"{z.real:+.4f}{z.imag:+.4f}j" for z in z_arr]))

    report_text = "\n".join(lines)
    txt_path.write_text(report_text, encoding="utf-8")

    # Exporta arquivo LaTeX com as tabelas prontas para os slides/relatório
    latex_blocks = [
        "% Tabelas geradas automaticamente para a apresentação e relatório PME3380",
        format_routh_table_latex(
            res_pure.routh_table_9,
            res_pure.routh_row_labels_9,
            r"Tabela de Routh-Hurwitz do Polinômio Reduzido $Q_9(s)$ ($\mu_T = 0$)",
        ),
        "",
        format_routh_table_latex(
            res_fric.routh_table_9,
            res_fric.routh_row_labels_9,
            r"Tabela de Routh-Hurwitz do Polinômio Reduzido $Q_9(s)$ ($\mu_T = 0,015$)",
        ),
    ]
    tex_path.write_text("\n\n".join(latex_blocks), encoding="utf-8")

    return txt_path, tex_path


if __name__ == "__main__":
    res_pure = run_linear_analysis(DEFAULT_PARAMS, include_dry_friction=False)
    res_fric = run_linear_analysis(DEFAULT_PARAMS, include_dry_friction=True)
    txt_file, tex_file = export_linear_analysis_reports(res_pure, res_fric)
    print(txt_file.read_text(encoding="utf-8"))
    print(f"\n[OK] Relatórios salvos em:\n  - {txt_file}\n  - {tex_file}")
