import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp

# Definir la función para convertir una cadena de entrada en una función evaluable
def parse_equation(equation_str):
    x, y = sp.symbols('x y')
    expr = sp.sympify(equation_str)
    f = sp.lambdify((x, y), expr, 'numpy')
    return f

# Función para graficar las isoclinas
def isoclines(f, x_range, y_range, c_values):
    X, Y = np.meshgrid(x_range, y_range)
    plt.figure(figsize=(12, 8))
    for c in c_values:
        Z = f(X, Y) - c
        plt.contour(X, Y, Z, levels=[0], colors='blue', linestyles='dotted')

# Función para dibujar el campo de direcciones
def direction_field(f, x_range, y_range, step=1):
    X, Y = np.meshgrid(np.arange(x_range[0], x_range[1], step),
                       np.arange(y_range[0], y_range[1], step))
    U = 1
    V = f(X, Y)
    N = np.sqrt(U**2 + V**2)
    U2, V2 = U/N, V/N
    plt.quiver(X, Y, U2, V2, angles='xy')

# Función para resolver y graficar la solución de la ODE
def solve_ode(f, x0, y0, x_range):
    def dydx(y, x):
        return f(x, y)
    x = np.linspace(x_range[0], x_range[1], 100)
    sol = odeint(dydx, y0, x)
    plt.plot(x, sol, label=f'Initial condition: ({x0},{y0})')

# Solicitar la entrada del usuario
equation_str = input("Ingrese la ecuación diferencial en términos de x e y (por ejemplo, x + y**2): ")

# Parsear la ecuación ingresada
f = parse_equation(equation_str)

# Definir los rangos y valores
x_range = (-10, 10)
y_range = (-10, 10)
c_values = np.linspace(-10, 10, 20)

# Graficar las isoclinas, el campo de direcciones y la solución de la ODE
isoclines(f, np.linspace(x_range[0], x_range[1], 100), np.linspace(y_range[0], y_range[1], 100), c_values)
direction_field(f, x_range, y_range)
solve_ode(f, 0, 0, x_range)

plt.title(f'y\' = {equation_str}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
