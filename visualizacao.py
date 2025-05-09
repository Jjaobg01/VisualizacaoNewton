import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

x1_sym, x2_sym = sp.symbols('x1 x2')

f_sym = (x1_sym + 2*x2_sym - 7)**2 + (2*x1_sym + x2_sym - 5)**2

grad_f = [sp.diff(f_sym, var) for var in (x1_sym, x2_sym)]
hess_f = sp.hessian(f_sym, (x1_sym, x2_sym))

f_num = sp.lambdify((x1_sym, x2_sym), f_sym, 'numpy')
grad_f_num = sp.lambdify((x1_sym, x2_sym), grad_f, 'numpy')
hess_f_num = sp.lambdify((x1_sym, x2_sym), hess_f, 'numpy')

x_test = (1.0, 3.0)
print("f(1, 3) =", f_num(*x_test))
print("grad_f(1, 3) =", grad_f_num(*x_test))
print("hess_f(1, 3) =\n", hess_f_num(*x_test))

x1 = np.linspace(-10, 10, 400)
x2 = np.linspace(-10, 10, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = f_num(X1, X2)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.9)
ax1.set_title('Booth Function - 3D')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('f(x1, x2)')

ax2 = fig.add_subplot(1, 2, 2)
contours = ax2.contour(X1, X2, Z, levels=50, cmap='viridis')
ax2.clabel(contours, inline=True, fontsize=8)
ax2.plot(1, 3, 'ro') 
ax2.set_title('Curvas de Nível - Booth')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')

plt.tight_layout()
plt.show()
