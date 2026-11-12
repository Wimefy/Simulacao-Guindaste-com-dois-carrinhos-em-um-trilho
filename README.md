# PME3380 — Modelagem 2D de um Guindaste de Torre com Dois Trolleys no Mesmo Trilho Segurando uma Plataforma com Carga

Projeto completo desenvolvido em Python para a disciplina **PME3380 — Modelagem de Sistemas Dinâmicos** da **Escola Politécnica da Universidade de São Paulo (Poli-USP)**.

---

## 1. Visão Geral e Hipóteses Físicas do Modelo

O sistema consiste em um guindaste de torre plano (2D) cujo trilho horizontal situa-se em $y = 0$ (com eixo $X$ horizontal ao longo da lança de comprimento $L_{\text{total}}$ e eixo $Y$ vertical apontando **para baixo**, no sentido da gravidade $g$).

1. **Dois Trolleys Idênticos ($m_T$):** Movem-se sobre o mesmo trilho em $x_1(t)$ e $x_2(t)$, acionados por forças independentes dos motores $F_{M1}(t)$ e $F_{M2}(t)$ e sujeitos a atrito viscoso ($b_T$) e atrito seco de rolamento regularizado ($\mu_T N_i \tanh(\dot{x}_i / 0{,}01)$).
2. **Cabos Viscoelásticos Unilaterais:** Possuem massa desprezível, comprimento natural igual $L_s$, rigidez elástica $k$ e amortecimento interno $c_s$, trabalhando exclusivamente sob tração:
   $$T_i = \max\bigl(0,\, k(l_i - L_s) + c_s \dot{l}_i\bigr), \quad i \in \{1, 2\}$$
3. **Corpo Rígido Composto (Plataforma + Carga):** O atrito entre a carga ($m_c$) e a plataforma ($m_p$) impede escorregamento relativo, formando um único corpo rígido de massa $M = m_p + m_c$ e momento de inércia baricêntrico:
   $$I_G = \frac{1}{12} m_p L_p^2 + \frac{1}{12} m_c \left(L_c^2 + h_c^2\right), \quad h_c = \frac{A_c}{L_c}$$
4. **Perturbação Aerodinâmica de Vento:** Atua horizontalmente sobre a área lateral da carga ($A_c$):
   $$F_{\text{vento}} = \frac{1}{2} \rho_{\text{ar}} C_d A_c (V_v - \dot{x}_p)\,|V_v - \dot{x}_p|$$

---

## 2. Estrutura Modular do Projeto

| Arquivo | Descrição |
| :--- | :--- |
| [`config.py`](config.py) | Dataclass `CraneParams` e dicionário configurável com todos os parâmetros físicos, geométricos e numéricos padrão. |
| [`model.py`](model.py) | Cinemática analítica exata ($l_i, \dot{l}_i, \theta_i, \dot{\theta}_i$), equações de estado não-lineares $\dot{\mathbf{x}}_n = \mathbf{f}(\mathbf{x}_n, \mathbf{u}_n, \mathbf{w}_n)$ (Ordem 10), saídas $\mathbf{y}_n$ (Ordem 7) e balanço de energia mecânica. |
| [`analysis_linear.py`](analysis_linear.py) | Verificação explícita do equilíbrio estático $\mathbf{f}(\mathbf{x}_{\text{eq}}, \mathbf{u}_{\text{eq}}, \mathbf{w}_{\text{eq}}) = \mathbf{0}$, linearização simbólica via `SymPy` ($\mathbf{A}, \mathbf{B}, \mathbf{E}, \mathbf{C}, \mathbf{D}$), análise modal ($\omega_n, \zeta$), **Tabela de Routh-Hurwitz** (ASCII e LaTeX) e **Funções de Transferência** (`control` / `scipy.signal`). |
| [`simulation.py`](simulation.py) | Integração numérica de alta precisão (`solve_ivp` com `RK45` e `Radau`, `rtol=1e-7, atol=1e-9`), estudo de convergência numérica e 3 cenários de operação. |
| [`plots.py`](plots.py) | Geração de gráficos científicos em alta resolução (300 DPI `.png` e `.pdf` vetorial) formatados para slides da apresentação PME3380. |
| [`animation.py`](animation.py) | Animação 2D fisicamente rica (`FuncAnimation`) com torre e lança treliçadas, contrapeso, 2 trolleys com rodas, cabos coloridos pela tração, plataforma + carga rotacionando por $\phi(t)$, setas dinâmicas de forças dos motores e vento, HUD de telemetria e painel lateral sincronizado. |
| [`main.py`](main.py) | Script principal que executa todo o pipeline e exporta relatórios, tabelas LaTeX, figuras e vídeos (`.mp4` e `.gif`). |

---

## 3. Como Executar

Utilizando o ambiente virtual `.venv` configurado no projeto:

```bash
# Executar todo o pipeline (linearização, Routh-Hurwitz, 3 cenários, gráficos e animações MP4/GIF):
.venv/bin/python main.py

# Executar apenas análise linear, simulações e gráficos estáticos (sem renderizar vídeos):
.venv/bin/python main.py --no-anim

# Executar e abrir janela interativa com a animação de um cenário específico:
.venv/bin/python main.py --scenario cenario3 --show
```

Todos os resultados são exportados automaticamente em:
- `outputs/tables/` — Relatório completo da linearização (`relatorio_linearizacao.txt`) e tabelas em LaTeX (`tabelas_latex_pme3380.tex`);
- `outputs/figures/` — Gráficos em alta resolução (`.png` e `.pdf`) e quadros-chave das animações;
- `outputs/animations/` — Animações 2D dos 3 cenários em `.mp4` e `.gif`.
