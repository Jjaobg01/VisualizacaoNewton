import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# variaveis globais
x1_sym, x2_sym = sp.symbols('x1 x2')

# funcao exemplo
f_simbolo = (x1_sym - 3)**2 + (x2_sym + 4)**2 

# ponto inicial (pode ser ajustado conforme a funcao)
x0 = [5, -5]

def preparar_funcao(f_sym):
    grad = [sp.diff(f_sym, var) for var in (x1_sym, x2_sym)]
    hess = sp.hessian(f_sym, (x1_sym, x2_sym))
    f_num = sp.lambdify((x1_sym, x2_sym), f_sym, 'numpy')
    grad_f = sp.lambdify((x1_sym, x2_sym), grad, 'numpy')
    hess_f = sp.lambdify((x1_sym, x2_sym), hess, 'numpy')
    return f_num, grad_f, hess_f

def obter_fator_passo(fator=1.0):
    return float(fator)

def metodo_newton(f_num, grad_f, hess_f, x0,
                  metodo='newton',
                  fator_passo=None,
                  tol=1e-6,
                  max_iter=100):
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
        try:
            H0 = np.array(hess_f(*x_k), dtype=np.float64)
            H_inv_const = np.linalg.inv(H0)
        except np.linalg.LinAlgError:
            raise ValueError("Hessiana inicial não é inversível no ponto inicial.")

    for k in range(max_iter):
        g_k = np.array(grad_f(*x_k), dtype=np.float64)

        if metodo == 'newton':
            try:
                H_k = np.array(hess_f(*x_k), dtype=np.float64)
                delta = -np.linalg.solve(H_k, g_k)
            except np.linalg.LinAlgError:
                print("Hessiana singular durante a iteração. Encerrando.")
                break
        else:
            delta = -H_inv_const.dot(g_k)

        alpha_k = fator_passo(k)
        x_k = x_k + alpha_k * delta
        historico.append(x_k.copy())

        if np.linalg.norm(alpha_k * delta) < tol:
            break

    return np.array(historico)

def calcular_limites(traj1, traj2, margem=0.5):
    todas_coords = np.vstack((traj1, traj2))
    x_min, x_max = todas_coords[:, 0].min(), todas_coords[:, 0].max()
    y_min, y_max = todas_coords[:, 1].min(), todas_coords[:, 1].max()
    return (x_min - margem, x_max + margem), (y_min - margem, y_max + margem)

def plot_3d(f_num, x_lim=(-5, 5), y_lim=(-5, 5), n_points=400, titulo='Função Objetivo'):
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
    x1_vals = np.linspace(x_lim[0], x_lim[1], n_points)
    x2_vals = np.linspace(y_lim[0], y_lim[1], n_points)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = f_num(X1, X2)

    traj1 = np.array(traj_newton)
    traj2 = np.array(traj_modificado)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.contour(X1, X2, Z, levels=50, cmap='viridis')
    ax1.plot(traj1[:, 0], traj1[:, 1], 'r-o', label='Newton')
    ax1.plot(traj1[0, 0], traj1[0, 1], 'ko', label='Início')
    ax1.plot(traj1[-1, 0], traj1[-1, 1], 'go', label='Final')
    ax1.set_title('Newton Clássico')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.legend()
    ax1.grid(True)

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
    fator_exemplo = obter_fator_passo(1.0)

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

    x_lim, y_lim = calcular_limites(traj_newton, traj_mod)

    plot_3d(f_num, x_lim=x_lim, y_lim=y_lim, n_points=400, titulo='Função Objetivo')
    plot_2d_duplo(f_num, traj_newton, traj_mod, x_lim=x_lim, y_lim=y_lim, n_points=400)
