"""
config.py - Parâmetros Configuráveis Padrão do Guindaste de Torre com Dois Trolleys.

Disciplina: PME3380 - Modelagem de Sistemas Dinâmicos (Poli-USP)
Projeto: Modelagem 2D de um Guindaste de Torre com Dois Trolleys no Mesmo Trilho
         segurando uma Plataforma com Carga.

Este módulo centraliza todos os parâmetros físicos, geométricos e numéricos do
sistema no dataclass `CraneParams` (e dicionário equivalente), permitindo fácil
alteração para análises paramétricas e simulações.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import os
from pathlib import Path
from typing import Any, Dict


# Diretórios padrão de saída de artefatos para os slides e relatório PME3380
BASE_DIR = Path(__file__).resolve().parent
if "MPLCONFIGDIR" not in os.environ:
    MPL_CACHE_DIR = BASE_DIR / ".mplconfig"
    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)
else:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
ANIMATIONS_DIR = OUTPUT_DIR / "animations"
TABLES_DIR = OUTPUT_DIR / "tables"

for _directory in (OUTPUT_DIR, FIGURES_DIR, ANIMATIONS_DIR, TABLES_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


@dataclass
class CraneParams:
    """
    Parâmetros físicos e geométricos padrão do Guindaste 2D com Dois Trolleys.

    Todos os valores estão no Sistema Internacional de Unidades (SI):
    [kg, m, s, N, rad].
    """

    # --- Massas ---
    m_T: float = 350.0        # Massa de cada trolley [kg]
    m_p: float = 450.0        # Massa da plataforma [kg]
    m_c: float = 2500.0       # Massa da carga [kg]

    # --- Dimensões Geométricas ---
    L_total: float = 50.0     # Comprimento total do trilho da lança [m]
    L_s: float = 15.0         # Comprimento sem esforço (natural) de cada cabo [m]
    L_p: float = 6.0          # Comprimento da plataforma [m]
    L_c: float = 4.0          # Comprimento da carga [m]
    A_c: float = 8.0          # Área lateral da carga exposta ao vento [m^2]

    # --- Propriedades Viscoelásticas dos Cabos ---
    k: float = 2.5e5          # Rigidez elástica de cada cabo [N/m]
    c_s: float = 1200.0       # Amortecimento interno de cada cabo [N.s/m]

    # --- Parâmetros de Contato e Atrito nos Trolleys ---
    mu_T: float = 0.015       # Coeficiente de atrito de rolamento trolley-trilho [-]
    b_T: float = 150.0        # Coeficiente de atrito viscoso equivalente do trolley [N.s/m]
    v_reg: float = 0.01       # Velocidade característica de regularização da tanh [m/s]

    # --- Parâmetros Aerodinâmicos e Ambientais ---
    C_d: float = 1.15         # Coeficiente de arrasto aerodinâmico da carga [-]
    rho_ar: float = 1.225     # Massa específica do ar [kg/m^3]
    g: float = 9.81           # Aceleração da gravidade [m/s^2]

    # --- Posição de Referência de Equilíbrio no Trilho ---
    x1_eq_ref: float = 10.0   # Posição horizontal de equilíbrio do Trolley 1 [m]

    # --- Propriedades Calculadas Automaticamente ---
    I_G: float = field(init=False)  # Momento de inércia baricêntrico composto [kg.m^2]

    def __post_init__(self) -> None:
        """Calcula automaticamente o momento de inércia baricêntrico I_G."""
        self.recompute_derived()

    def recompute_derived(self) -> None:
        """
        Atualiza `I_G` de acordo com a expressão exigida:
        I_G = (1/12)*m_p*(L_p^2) + (1/12)*m_c*(L_c^2 + (A_c/L_c)^2)
        """
        h_c = self.A_c / self.L_c
        self.I_G = (1.0 / 12.0) * self.m_p * (self.L_p ** 2) + (1.0 / 12.0) * self.m_c * (
            self.L_c ** 2 + h_c ** 2
        )

    @property
    def M(self) -> float:
        """Massa total do corpo rígido composto (plataforma + carga) [kg]."""
        return self.m_p + self.m_c

    @property
    def h_c(self) -> float:
        """Altura lateral da carga [m], dada por h_c = A_c / L_c."""
        return self.A_c / self.L_c

    @property
    def T_eq(self) -> float:
        """Tração estática de equilíbrio em cada cabo [N]."""
        return 0.5 * self.M * self.g

    @property
    def delta_l_eq(self) -> float:
        """Alongamento estático de equilíbrio de cada cabo [m]."""
        return self.T_eq / self.k

    @property
    def y_p_eq(self) -> float:
        """Coordenada vertical de equilíbrio da plataforma (e comprimento l_eq do cabo) [m]."""
        return self.L_s + (self.M * self.g) / (2.0 * self.k)

    def copy_with(self, **kwargs: Any) -> "CraneParams":
        """Retorna uma nova instância de `CraneParams` com parâmetros específicos alterados."""
        new_instance = replace(self, **kwargs)
        new_instance.recompute_derived()
        return new_instance

    def to_dict(self) -> Dict[str, float]:
        """Exporta todos os parâmetros (incluindo derivados) como um dicionário Python."""
        d = asdict(self)
        d["M"] = self.M
        d["h_c"] = self.h_c
        d["T_eq"] = self.T_eq
        d["delta_l_eq"] = self.delta_l_eq
        d["y_p_eq"] = self.y_p_eq
        return d


# Instância padrão global e dicionário facilmente editável
DEFAULT_PARAMS = CraneParams()
CRANE_PARAMS_DICT: Dict[str, float] = DEFAULT_PARAMS.to_dict()

# Configurações numéricas padrão para integração de EDOs (scipy.integrate.solve_ivp)
SIMULATION_CONFIG: Dict[str, Any] = {
    "method": "RK45",
    "rtol": 1e-7,
    "atol": 1e-9,
    "max_step": 0.02,
    "fps_animation": 30,
    "dpi_figures": 300,
}
