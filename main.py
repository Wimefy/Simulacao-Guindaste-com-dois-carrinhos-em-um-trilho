"""
main.py - Ponto de Entrada Principal do Projeto PME3380 (Poli-USP).

Executa de forma integrada:
  1. Verificação de Equilíbrio Estático e Linearização Simbólica (SymPy);
  2. Polinômio Característico, Autovalores/Modos Naturais, Tabela de Routh-Hurwitz e
     Funções de Transferência (exportando relatórios em texto e LaTeX);
  3. Simulação Numérica Não-Linear (`solve_ivp` RK45/Radau) dos 3 Cenários + Teste de Convergência;
  4. Geração de todos os Gráficos Científicos em alta resolução (PNG 300 DPI + PDF vetorial);
  5. Geração das Animações 2D fisicamente detalhadas (`.mp4` e `.gif`).
"""

from __future__ import annotations

import argparse
from typing import List

from config import ANIMATIONS_DIR, DEFAULT_PARAMS, FIGURES_DIR, TABLES_DIR
from analysis_linear import export_linear_analysis_reports, run_linear_analysis
from simulation import run_all_scenarios, run_convergence_study
from plots import generate_all_plots
from animation import create_crane_animation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PME3380 - Modelagem 2D de Guindaste de Torre com Dois Trolleys e Plataforma com Carga"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["all", "cenario1", "cenario2", "cenario3"],
        help="Cenário para animar (padrão: 'all' gera animações para todos os 3 cenários).",
    )
    parser.add_argument(
        "--no-anim",
        action="store_true",
        help="Pula a renderização dos vídeos .mp4/.gif (executa apenas análise linear, simulação e gráficos).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Abre janela interativa do Matplotlib para visualizar a animação em tempo real.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Passo de subamostragem de frames na animação para acelerar a exportação (padrão: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = DEFAULT_PARAMS

    print("=" * 88)
    print(" INICIANDO PROJETO PME3380 - GUINDASTE DE TORRE 2D COM DOIS TROLLEYS")
    print("=" * 88)

    # 1. Análise de Equilíbrio e Linearização
    print("\n[1/4] Executando análise simbólica de equilíbrio, Jacobianas, Routh-Hurwitz e FTs...")
    res_pure = run_linear_analysis(params, include_dry_friction=False)
    res_fric = run_linear_analysis(params, include_dry_friction=True)
    txt_report, tex_report = export_linear_analysis_reports(res_pure, res_fric, TABLES_DIR)
    print(txt_report.read_text(encoding="utf-8"))
    print(f"  -> Relatório salvo em: {txt_report}")
    print(f"  -> Tabelas LaTeX salvas em: {tex_report}")

    # 2. Simulação Numérica dos 3 Cenários + Estudo de Convergência
    print("\n[2/4] Executando simulações numéricas não-lineares (RK45, rtol=1e-7, atol=1e-9)...")
    lin_mats_fric = (res_fric.A, res_fric.B, res_fric.C, res_fric.D)
    scenarios = run_all_scenarios(params, linear_matrices=lin_mats_fric)

    for key, scen in scenarios.items():
        min_T1 = float(scen.T1.min())
        min_T2 = float(scen.T2.min())
        max_th1 = float( abs(scen.y[0]).max() * 180.0 / 3.141592653589793 )
        max_phi = float( abs(scen.x[8]).max() * 180.0 / 3.141592653589793 )
        print(
            f"  -> {scen.title}: concluído em {scen.cpu_time_s:.3f} s ({scen.nfev} aval.) | "
            f"T_min = ({min_T1/1000:.2f}, {min_T2/1000:.2f}) kN > 0 | "
            f"|theta1|_max = {max_th1:.2f}° | |phi|_max = {max_phi:.3f}°"
        )

    print("\n  Executando teste de convergência numérica (RK45 vs Radau)...")
    conv_results = run_convergence_study(params)
    for row in conv_results:
        print(
            f"    Método {row['method']:5s} | rtol={row['rtol']:.0e} | "
            f"Erro_inf={row['err_inf']:.3e} | Erro_RMS={row['err_rms']:.3e} | "
            f"nfev={row['nfev']:6d} | t_CPU={row['cpu_time_s']:.3f} s"
        )

    # 3. Gráficos Científicos para Slides
    print("\n[3/4] Gerando gráficos científicos em alta resolução (PNG 300 DPI + PDF)...")
    saved_figs = generate_all_plots(scenarios, res_pure, res_fric, conv_results, FIGURES_DIR)
    print(f"  -> {len(saved_figs)} arquivos gráficos gerados em: {FIGURES_DIR}")

    # 4. Animação 2D
    if not args.no_anim:
        print("\n[4/4] Renderizando animações 2D (.mp4, .gif e quadros-chave .png)...")
        target_keys: List[str] = (
            list(scenarios.keys()) if args.scenario == "all" else [args.scenario]
        )
        for k_scen in target_keys:
            scen_obj = scenarios[k_scen]
            print(f"  -> Renderizando animação de '{scen_obj.title}'...")
            anim_paths = create_crane_animation(
                scen_obj,
                output_dir=ANIMATIONS_DIR,
                fps=25,
                frame_stride=args.stride,
                save_mp4=True,
                save_gif=True,
                show_interactive=args.show,
            )
            for fmt, path_val in anim_paths.items():
                print(f"     [{fmt.upper()}] {path_val}")

    print("\n" + "=" * 88)
    print(" PIPELINE CONCLUÍDO COM SUCESSO!")
    print("=" * 88)


if __name__ == "__main__":
    main()

