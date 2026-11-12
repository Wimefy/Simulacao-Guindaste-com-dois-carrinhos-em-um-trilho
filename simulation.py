"""
simulation.py - Simulação Numérica Não-Linear, Cenários de Operação e Estudo de Convergência.

Disciplina: PME3380 - Modelagem de Sistemas Dinâmicos (Poli-USP)
Projeto: Modelagem 2D de um Guindaste de Torre com Dois Trolleys no Mesmo Trilho
         segurando uma Plataforma com Carga.

Cenários Implementados:
  - Cenário 1: Oscilação Livre (Condição Inicial fora do equilíbrio) + Comparação Linear vs Não-Linear
               e Balanço de Energia Mecânica (Conservação / Dissipação).
  - Cenário 2: Translação Sincronizada com Frenagem (F_M1(t) = F_M2(t)).
  - Cenário 3: Perturbação Severa de Rajada de Vento + Assimetria nos Motores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, Dict, List, Tuple
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

from config import CraneParams, DEFAULT_PARAMS, SIMULATION_CONFIG
from model import (
    compute_energy,
    compute_kinematics_and_forces,
    f_nonlinear,
    get_equilibrium,
)


@dataclass
class SimulationResult:
    """Armazena todas as séries temporais de um cenário de simulação."""

    scenario_id: str
    title: str
    description: str
    params: CraneParams
    t: np.ndarray                 # Vetor de tempo [s] (N,)
    x: np.ndarray                 # Estados não-lineares (10, N)
    u: np.ndarray                 # Entradas de controle [F_M1, F_M2] (2, N)
    w: np.ndarray                 # Perturbação de vento [V_v] (1, N)
    y: np.ndarray                 # Saídas [theta1, dtheta1, theta2, dtheta2, x_p, y_p, phi] (7, N)
    T1: np.ndarray                # Tração no cabo 1 [N] (N,)
    T2: np.ndarray                # Tração no cabo 2 [N] (N,)
    l1: np.ndarray                # Comprimento do cabo 1 [m] (N,)
    l2: np.ndarray                # Comprimento do cabo 2 [m] (N,)
    dl1: np.ndarray               # Taxa de alongamento do cabo 1 [m/s] (N,)
    dl2: np.ndarray               # Taxa de alongamento do cabo 2 [m/s] (N,)
    x_A: np.ndarray               # Ponto de ancoragem A - coord X [m] (N,)
    y_A: np.ndarray               # Ponto de ancoragem A - coord Y [m] (N,)
    x_B: np.ndarray               # Ponto de ancoragem B - coord X [m] (N,)
    y_B: np.ndarray               # Ponto de ancoragem B - coord Y [m] (N,)
    F_vento: np.ndarray           # Força aerodinâmica horizontal [N] (N,)
    E_cin: np.ndarray             # Energia cinética [J] (N,)
    E_pot: np.ndarray             # Energia potencial (grav + elástica) [J] (N,)
    E_total: np.ndarray           # Energia mecânica total instantânea [J] (N,)
    P_dissip: np.ndarray          # Potência instantânea dissipada [W] (N,)
    E_dissip_acum: np.ndarray     # Energia dissipada acumulada [J] (N,)
    cpu_time_s: float = 0.0
    nfev: int = 0
    x_linear: np.ndarray | None = field(default=None)
    y_linear: np.ndarray | None = field(default=None)


def smooth_step(t: float | np.ndarray, t_on: float, t_off: float, rise: float = 0.35) -> float | np.ndarray:
    """Pulso retangular suavizado via tangente hiperbólica (C^infinito) entre t_on e t_off."""
    return 0.5 * (np.tanh((t - t_on) / rise) - np.tanh((t - t_off) / rise))


# ==============================================================================
# Perfis de Entradas u(t) e Perturbações w(t) dos 3 Cenários
# ==============================================================================

def inputs_scenario_1(t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Cenário 1: Oscilação livre (sem forças de motor e sem vento)."""
    return np.array([0.0, 0.0], dtype=float), np.array([0.0], dtype=float)


def inputs_scenario_2(t: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cenário 2: Translação Sincronizada com Frenagem (F_M1(t) = F_M2(t)).
      - Aceleração sincronizada de t = 1.0 s a t = 6.0 s (+1650 N em cada motor)
      - Cruzeiro compensando atrito de t = 6.0 s a t = 12.5 s (+295 N em cada motor)
      - Frenagem sincronizada de t = 12.5 s a t = 17.5 s (-1420 N em cada motor)
      - Repouso dos motores t > 17.5 s para observar a oscilação pendular residual.
    """
    F_acc = 1650.0 * smooth_step(t, 1.0, 6.0, rise=0.35)
    F_cruise = 295.0 * smooth_step(t, 6.0, 12.5, rise=0.35)
    F_brake = -1420.0 * smooth_step(t, 12.5, 17.5, rise=0.35)
    F_sync = float(F_acc + F_cruise + F_brake)
    return np.array([F_sync, F_sync], dtype=float), np.array([0.0], dtype=float)


def inputs_scenario_3(t: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cenário 3: Perturbação Severa de Rajada de Vento + Assimetria nos Motores.
      - Motores executam manobra com assimetria (reposicionamento até o meio da lança
        e compensação diferencial de espaçamento entre trolleys);
      - Rajada severa de vento horizontal V_v(t) entre t = 3.0 s e t = 25.0 s:
        degrau suavizado de 11.5 m/s + rajadas turbulentas harmônicas atingindo ~18.5 m/s.
    """
    # Atuação assimétrica dos motores mantendo trolleys dentro do trilho (x in [10, 38] m)
    F_M1 = (
        1250.0 * smooth_step(t, 1.0, 6.0, rise=0.35)
        - 1150.0 * smooth_step(t, 11.5, 16.5, rise=0.35)
        + 420.0 * smooth_step(t, 6.0, 22.0, rise=0.40) * np.sin(0.65 * (t - 6.0))
    )
    F_M2 = (
        1450.0 * smooth_step(t, 1.0, 6.0, rise=0.35)
        - 1350.0 * smooth_step(t, 11.5, 16.5, rise=0.35)
        - 420.0 * smooth_step(t, 6.0, 22.0, rise=0.40) * np.sin(0.65 * (t - 6.0))
    )

    # Perfil de vento severo: degrau suavizado + rajadas senoidais compostas
    envelope_gust = smooth_step(t, 3.0, 25.0, rise=0.60)
    gust_osc = (
        4.5 * np.sin(2.0 * np.pi * 0.28 * (t - 3.0))
        + 2.8 * np.sin(2.0 * np.pi * 0.72 * (t - 3.0) + 0.6)
    )
    V_v = float(envelope_gust * (11.5 + gust_osc))
    return np.array([float(F_M1), float(F_M2)], dtype=float), np.array([V_v], dtype=float)


def simulate_system(
    scenario_id: str,
    title: str,
    description: str,
    x0: np.ndarray,
    input_func: Callable[[float], Tuple[np.ndarray, np.ndarray]],
    t_span: Tuple[float, float],
    n_points: int = 1051,
    params: CraneParams = DEFAULT_PARAMS,
    method: str = SIMULATION_CONFIG["method"],
    rtol: float = SIMULATION_CONFIG["rtol"],
    atol: float = SIMULATION_CONFIG["atol"],
    max_step: float = SIMULATION_CONFIG["max_step"],
    linear_matrices: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> SimulationResult:
    """
    Integra numericamente o sistema não-linear de ordem 10 usando `scipy.integrate.solve_ivp`
    e reconstrói todas as variáveis cinemáticas, forças, trações e energias.
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    def ode_rhs(t: float, x: np.ndarray) -> np.ndarray:
        u_t, w_t = input_func(t)
        return f_nonlinear(t, x, u_t, w_t, params)

    t_start = time.perf_counter()
    sol = solve_ivp(
        fun=ode_rhs,
        t_span=t_span,
        y0=x0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    cpu_time = time.perf_counter() - t_start

    if not sol.success:
        raise RuntimeError(f"Falha na integração numérica do cenário {scenario_id}: {sol.message}")

    t_arr = sol.t
    x_arr = sol.y
    N = len(t_arr)

    u_arr = np.zeros((2, N), dtype=float)
    w_arr = np.zeros((1, N), dtype=float)
    y_arr = np.zeros((7, N), dtype=float)
    T1_arr = np.zeros(N, dtype=float)
    T2_arr = np.zeros(N, dtype=float)
    l1_arr = np.zeros(N, dtype=float)
    l2_arr = np.zeros(N, dtype=float)
    dl1_arr = np.zeros(N, dtype=float)
    dl2_arr = np.zeros(N, dtype=float)
    xA_arr = np.zeros(N, dtype=float)
    yA_arr = np.zeros(N, dtype=float)
    xB_arr = np.zeros(N, dtype=float)
    yB_arr = np.zeros(N, dtype=float)
    F_vento_arr = np.zeros(N, dtype=float)
    E_cin_arr = np.zeros(N, dtype=float)
    E_pot_arr = np.zeros(N, dtype=float)
    E_tot_arr = np.zeros(N, dtype=float)
    P_diss_arr = np.zeros(N, dtype=float)

    for k_idx in range(N):
        tk = float(t_arr[k_idx])
        xk = x_arr[:, k_idx]
        uk, wk = input_func(tk)
        u_arr[:, k_idx] = uk
        w_arr[:, k_idx] = wk

        kin = compute_kinematics_and_forces(xk, uk, wk, params)
        y_arr[:, k_idx] = [
            kin["theta1"],
            kin["dtheta1"],
            kin["theta2"],
            kin["dtheta2"],
            xk[4],
            xk[6],
            xk[8],
        ]
        T1_arr[k_idx] = kin["T1"]
        T2_arr[k_idx] = kin["T2"]
        l1_arr[k_idx] = kin["l1"]
        l2_arr[k_idx] = kin["l2"]
        dl1_arr[k_idx] = kin["dl1"]
        dl2_arr[k_idx] = kin["dl2"]
        xA_arr[k_idx] = kin["x_A"]
        yA_arr[k_idx] = kin["y_A"]
        xB_arr[k_idx] = kin["x_B"]
        yB_arr[k_idx] = kin["y_B"]
        F_vento_arr[k_idx] = kin["F_vento"]

        en = compute_energy(xk, params)
        E_cin_arr[k_idx] = en["E_cin"]
        E_pot_arr[k_idx] = en["E_pot"]
        E_tot_arr[k_idx] = en["E_total"]
        P_diss_arr[k_idx] = en["P_dissip"]

    E_dissip_acum = cumulative_trapezoid(P_diss_arr, t_arr, initial=0.0)

    # Integração opcional do modelo linearizado em torno de x_eq para comparação direta
    x_lin_arr = None
    y_lin_arr = None
    if linear_matrices is not None:
        A_mat, B_mat, C_mat, D_mat = linear_matrices
        x_eq, u_eq, _, y_eq = get_equilibrium(params)
        dx0 = x0 - x_eq

        def lin_rhs(t: float, dx: np.ndarray) -> np.ndarray:
            u_t, _ = input_func(t)
            du = u_t - u_eq
            return A_mat @ dx + B_mat @ du

        sol_lin = solve_ivp(
            fun=lin_rhs,
            t_span=t_span,
            y0=dx0,
            method="RK45",
            t_eval=t_arr,
            rtol=1e-7,
            atol=1e-9,
        )
        x_lin_arr = sol_lin.y + x_eq[:, None]
        y_lin_arr = (C_mat @ sol_lin.y) + y_eq[:, None]

    return SimulationResult(
        scenario_id=scenario_id,
        title=title,
        description=description,
        params=params,
        t=t_arr,
        x=x_arr,
        u=u_arr,
        w=w_arr,
        y=y_arr,
        T1=T1_arr,
        T2=T2_arr,
        l1=l1_arr,
        l2=l2_arr,
        dl1=dl1_arr,
        dl2=dl2_arr,
        x_A=xA_arr,
        y_A=yA_arr,
        x_B=xB_arr,
        y_B=yB_arr,
        F_vento=F_vento_arr,
        E_cin=E_cin_arr,
        E_pot=E_pot_arr,
        E_total=E_tot_arr,
        P_dissip=P_diss_arr,
        E_dissip_acum=E_dissip_acum,
        cpu_time_s=cpu_time,
        nfev=int(sol.nfev),
        x_linear=x_lin_arr,
        y_linear=y_lin_arr,
    )


def run_all_scenarios(
    params: CraneParams = DEFAULT_PARAMS,
    linear_matrices: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> Dict[str, SimulationResult]:
    """Executa os 3 cenários padrões exigidos pelo projeto PME3380."""
    x_eq, _, _, _ = get_equilibrium(params)

    # --- Cenário 1: Oscilação Livre (Condição Inicial fora do equilíbrio) ---
    # Deslocamento lateral dx_p = +0.25 m, vertical dy_p = +15 mm e inclinação phi = +0.45 graus
    # (mantendo ambos os cabos estritamente tracionados T1, T2 > 0 e operando no regime linearizável!)
    x0_scen1 = x_eq.copy()
    x0_scen1[4] += 0.25                   # x_p(0) = x_p_eq + 0.25 m
    x0_scen1[6] += 0.015                  # y_p(0) = y_p_eq + 0.015 m
    x0_scen1[8] += np.deg2rad(0.45)       # phi(0) = +0.45 deg

    res1 = simulate_system(
        scenario_id="cenario1",
        title="Cenário 1: Oscilação Livre Fora do Equilíbrio",
        description=(
            "Trolleys inicialmente em repouso (x1=10m, x2=16m) e plataforma deslocada "
            "lateralmente (+0.25 m), verticalmente (+15 mm) e inclinada (+0.45°)."
        ),
        x0=x0_scen1,
        input_func=inputs_scenario_1,
        t_span=(0.0, 35.0),
        n_points=1051,
        params=params,
        linear_matrices=linear_matrices,
    )

    # --- Cenário 2: Translação Sincronizada com Frenagem ---
    res2 = simulate_system(
        scenario_id="cenario2",
        title="Cenário 2: Translação Sincronizada com Frenagem",
        description=(
            "Forças sincronizadas F_M1(t) = F_M2(t) acelerando os dois trolleys ao longo "
            "da lança, seguidas de cruzeiro e frenagem suave, excitando o modo pendular."
        ),
        x0=x_eq.copy(),
        input_func=inputs_scenario_2,
        t_span=(0.0, 35.0),
        n_points=1051,
        params=params,
        linear_matrices=linear_matrices,
    )

    # --- Cenário 3: Perturbação Severa de Rajada de Vento + Assimetria ---
    res3 = simulate_system(
        scenario_id="cenario3",
        title="Cenário 3: Rajada Severa de Vento + Assimetria nos Motores",
        description=(
            "Operação sob rajada severa de vento V_v(t) (degrau suavizado + rajadas "
            "harmônicas até 18.5 m/s) combinada com atuação assimétrica dos motores."
        ),
        x0=x_eq.copy(),
        input_func=inputs_scenario_3,
        t_span=(0.0, 35.0),
        n_points=1051,
        params=params,
        linear_matrices=linear_matrices,
    )

    return {
        "cenario1": res1,
        "cenario2": res2,
        "cenario3": res3,
    }


def run_convergence_study(
    params: CraneParams = DEFAULT_PARAMS,
) -> List[Dict[str, float | str]]:
    """
    Executa estudo de convergência numérica comparando diferentes métodos (`RK45` e `Radau`),
    tolerâncias relativas (`rtol` de 1e-3 a 1e-10) e passos máximos (`max_step`) frente a uma
    solução de referência de altíssima precisão (`DOP853`, `rtol=1e-12, atol=1e-14`).
    """
    x_eq, _, _, _ = get_equilibrium(params)
    x0 = x_eq.copy()
    x0[4] += 1.20
    x0[6] += 0.035
    x0[8] += np.deg2rad(3.5)

    t_span = (0.0, 15.0)
    t_eval = np.linspace(t_span[0], t_span[1], 301)

    def ode_rhs(t: float, x: np.ndarray) -> np.ndarray:
        u_t, w_t = inputs_scenario_3(t)
        return f_nonlinear(t, x, u_t, w_t, params)

    # Solução de referência de altíssima ordem (Dormand-Prince 8(5,3))
    sol_ref = solve_ivp(
        fun=ode_rhs,
        t_span=t_span,
        y0=x0,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-12,
        atol=1e-14,
        max_step=0.005,
    )
    x_ref = sol_ref.y

    test_configs = [
        ("RK45", 1e-3, 1e-5, 0.20),
        ("RK45", 1e-5, 1e-7, 0.05),
        ("RK45", 1e-7, 1e-9, 0.02),
        ("RK45", 1e-9, 1e-11, 0.01),
        ("Radau", 1e-3, 1e-5, 0.20),
        ("Radau", 1e-5, 1e-7, 0.05),
        ("Radau", 1e-7, 1e-9, 0.02),
        ("Radau", 1e-9, 1e-11, 0.01),
    ]

    results: List[Dict[str, float | str]] = []
    for method, rtol, atol, max_step in test_configs:
        t0 = time.perf_counter()
        sol = solve_ivp(
            fun=ode_rhs,
            t_span=t_span,
            y0=x0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        elapsed = time.perf_counter() - t0
        err_inf = float(np.max(np.abs(sol.y - x_ref)))
        err_rms = float(np.sqrt(np.mean((sol.y - x_ref) ** 2)))
        results.append(
            {
                "method": method,
                "rtol": rtol,
                "atol": atol,
                "max_step": max_step,
                "err_inf": max(err_inf, 1e-15),
                "err_rms": max(err_rms, 1e-15),
                "nfev": int(sol.nfev),
                "cpu_time_s": elapsed,
            }
        )

    return results
