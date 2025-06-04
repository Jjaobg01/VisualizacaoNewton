import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


# variaveis globais
x1_sym, x2_sym = sp.symbols('x1 x2')


# funcao de ex ( TROCAR PARA RECEBER FUNCAO DADA PELO USER !!!! )
f_simbolo = (1.5 - x1_sym + x1_sym * x2_sym)**2 + \
            (2.25 - x1_sym + x1_sym * x2_sym**2)**2 + \
            (2.625 - x1_sym + x1_sym * x2_sym**3)**2

# ponto inicial
x0 = [1.0, 3.0]

def preparar_funcao(f_sym):
    """
    Recebe uma expressão simbólica em x1, x2 e retorna:
      1) f_num(x1, x2)       – função numérica
      2) grad_f(x1, x2)      – gradiente numérico (array de dimensão 2)
      3) hess_f(x1, x2)      – hessiana numérica (matriz 2×2)
    """

    grad = [sp.diff(f_sym, var) for var in (x1_sym, x2_sym)]

    hess = sp.hessian(f_sym, (x1_sym, x2_sym))

    f_num = sp.lambdify((x1_sym, x2_sym), f_sym, 'numpy')
    grad_f = sp.lambdify((x1_sym, x2_sym), grad, 'numpy')
    hess_f = sp.lambdify((x1_sym, x2_sym), hess, 'numpy')
    return f_num, grad_f, hess_f


# funcao que obtem o fator de passo
def obter_fator_passo(fator=1.0):
    """
    Retorna o fator de passo (step size) para o método de Newton.
    Por padrão, retorna 1.0. Se quiser usar passo diferente, chame:
        obter_fator_passo(0.5)   # exemplo de passo = 0.5
    """
    return float(fator)


# MÉTODO DE NEWTON / NEWTON MODIFICADO GENERALIZADO

def metodo_newton(f_num, grad_f, hess_f, x0,
                  metodo='newton',
                  fator_passo=None,
                  tol=1e-6,
                  max_iter=100):
    """
    Executa o Método de Newton (ou sua versão modificada) em 2 variáveis.

    Parâmetros:
    - f_num: função numérica f(x1, x2)
    - grad_f: função que retorna [∂f/∂x1, ∂f/∂x2] em (x1, x2)
    - hess_f: função que retorna [[∂²f/∂x1², ∂²f/∂x1∂x2],
                                  [∂²f/∂x2∂x1, ∂²f/∂x2²]] em (x1, x2)
    - x0: lista ou array com 2 elementos, ponto inicial [x1_0, x2_0]
    - metodo: 'newton' ou 'modificado'
    - fator_passo: functor ou valor escalar. Se None, usa obter_fator_passo(1.0).
                   Se for escalar, será interpretado como passo constante.
                   Se for callable, deve retornar um float para cada iteração.
    - tol: tolerância para convergência ‖Δx‖ < tol
    - max_iter: número máximo de iterações

    Retorna:
    - historico: array de forma (it+1, 2) com os pontos x^(k)
    """
    # Se não vier fator_passo, use 1.0 sempre
    if fator_passo is None:
        fator_passo = obter_fator_passo(1.0)

    if not callable(fator_passo):
        constante = float(fator_passo)
        def _passo_iter(_k):
            return constante
        fator_passo = _passo_iter

    x_k = np.array(x0, dtype=np.float64)
    historico = [x_k.copy()]

    if metodo == 'modificado':
        H0 = np.array(hess_f(*x_k), dtype=np.float64)
        H_inv_const = np.linalg.inv(H0)

    for k in range(max_iter):
        g_k = np.array(grad_f(*x_k), dtype=np.float64)

        if metodo == 'newton':
            H_k = np.array(hess_f(*x_k), dtype=np.float64)
            try:
                delta = -np.linalg.solve(H_k, g_k)
            except np.linalg.LinAlgError:
                # se hessiana for singular, break
                break
        else:  # 'modificado'
            delta = -H_inv_const.dot(g_k)

        # obter passo
        alpha_k = fator_passo(k)
        x_k = x_k + alpha_k * delta

        historico.append(x_k.copy())

        if np.linalg.norm(alpha_k * delta) < tol:
            break

    return np.array(historico)


# funcoes pro plot 3d e 2d
def plot_3d(f_num, x_lim=(-5, 5), y_lim=(-5, 5), n_points=400, titulo='Função Objetivo'):
    """
    Plota a superfície 3D de f_num em uma malha retangular.
    - x_lim, y_lim definem o intervalo em x1 e x2 (tupla)
    - n_points: resolução da malha em cada direção
    """
    x1_vals = np.linspace(x_lim[0], x_lim[1], n_points)
    x2_vals = np.linspace(y_lim[0], y_lim[1], n_points)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = f_num(X1, X2)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.9)
    ax.set_title(f'{titulo} - Superfície 3D')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    plt.tight_layout()
    plt.show()

def plot_2d_duplo(f_num, traj_newton, traj_modificado,
                  x_lim=(-5, 5), y_lim=(-5, 5), n_points=400):
    """
    Plota dois gráficos 2D lado a lado:
      - Curvas de nível + trajetória do Newton clássico
      - Curvas de nível + trajetória do Newton modificado
    """
    x1_vals = np.linspace(x_lim[0], x_lim[1], n_points)
    x2_vals = np.linspace(y_lim[0], y_lim[1], n_points)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = f_num(X1, X2)

    traj1 = np.array(traj_newton)
    traj2 = np.array(traj_modificado)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


    # Newton Clássico (esquerda)
    ax1.contour(X1, X2, Z, levels=50, cmap='viridis')
    ax1.plot(traj1[:, 0], traj1[:, 1], 'r-o', label='Newton')
    ax1.plot(traj1[0, 0], traj1[0, 1], 'ko', label='Início')
    ax1.plot(traj1[-1, 0], traj1[-1, 1], 'go', label='Final')
    ax1.set_title('Newton Clássico')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.legend()
    ax1.grid(True)

    # newton modificado, aparece na direita
    ax2.contour(X1, X2, Z, levels=50, cmap='viridis')
    ax2.plot(traj2[:, 0], traj2[:, 1], 'b--o', label='Modificado')
    ax2.plot(traj2[0, 0], traj2[0, 1], 'ko', label='Início')
    ax2.plot(traj2[-1, 0], traj2[-1, 1], 'go', label='Final')
    ax2.set_title('Newton Modificado')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# execucao
if __name__ == "__main__":
    f_num, grad_f, hess_f = preparar_funcao(f_simbolo)

    # definir fator de passo (por exemplo, 1.0 para passo cheio)
    fator_exemplo = obter_fator_passo(1.0)

    # trajetorias
    traj_newton = metodo_newton(
        f_num, grad_f, hess_f, x0,
        metodo='newton',
        fator_passo=fator_exemplo,
        tol=1e-6,
        max_iter=100
    )

    traj_mod = metodo_newton(
        f_num, grad_f, hess_f, x0,
        metodo='modificado',
        fator_passo=fator_exemplo,
        tol=1e-6,
        max_iter=100
    )

    # lot 3D
    plot_3d(f_num, x_lim=(-5, 5), y_lim=(-5, 5), n_points=400, titulo='Função Objetivo')

    # plot 2D comparativo (Newton e Newton Modificado)
    plot_2d_duplo(f_num, traj_newton, traj_mod,
                  x_lim=(-5, 5), y_lim=(-5, 5), n_points=400)
