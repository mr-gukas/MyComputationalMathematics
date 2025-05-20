import numpy as np
import matplotlib.pyplot as plt

a = 1.0
X = 10.0
T = 0.5 * X / a

def phi(x):
    return np.sin(np.pi * x / 5)**4

def psi(t):
    return phi(-a * t)

def f(t, x):
    return 0.0

def analytical_solution(t, x):
    shifted_x = (x - a * t) % X
    return phi(shifted_x)

def corner_scheme(K, M):
    tau = T / K  
    h = X / M    
    
    if a * tau / h > 1:
        print(f"Warning: Scheme is unstable! CFL = {a * tau / h} > 1")
    
    u = np.zeros((K+1, M+1))
    
    for m in range(M+1):
        u[0, m] = phi(m * h)
    
    for k in range(K+1):
        u[k, 0] = u[k, M]
    
    for k in range(K):
        for m in range(1, M+1):
            u[k+1, m] = u[k, m] - a * tau / h * (u[k, m] - u[k, m-1]) + tau * f(k * tau, m * h)
        
        u[k+1, 0] = u[k+1, M]
    
    return u

def rectangle_scheme(K, M):
    tau = T / K  
    h = X / M    
    
    u = np.zeros((K+1, M+1))
    
    for m in range(M+1):
        u[0, m] = phi(m * h)
    
    u[0, 0] = u[0, M]
    
    sigma = a * tau / (4 * h)
    A = np.zeros((M, M))
    
    for i in range(M):
        A[i, i] = 1
    
    for i in range(M-1):
        A[i, i+1] = sigma
    A[M-1, 0] = sigma
    
    for i in range(1, M):
        A[i, i-1] = -sigma
    A[0, M-1] = -sigma
    
    for k in range(K):
        b = np.zeros(M)
        
        for m in range(M):
            m_prev = (m-1) % M
            m_next = (m+1) % M
            b[m] = u[k, m] + sigma * (u[k, m_next] - u[k, m_prev]) + tau * f(k * tau, m * h)
        
        u_interior = np.linalg.solve(A, b)
        
        u[k+1, 0:M] = u_interior
        
        u[k+1, M] = u[k+1, 0]
    
    return u

def lax_wendroff_scheme(K, M):
    tau = T / K  
    h = X / M    
    
    if a * tau / h > 1:
        print(f"Warning: Scheme may be unstable! CFL = {a * tau / h} > 1")
    
    u = np.zeros((K+1, M+1))
    
    for m in range(M+1):
        u[0, m] = phi(m * h)
    
    u[0, 0] = u[0, M]
    
    for k in range(K):
        for m in range(1, M+1):
            m_prev = (m-1) % (M+1)
            m_next = (m+1) % (M+1)
            
            u[k+1, m] = u[k, m] - 0.5 * a * tau / h * (u[k, m_next] - u[k, m_prev]) + \
                       0.5 * (a * tau / h)**2 * (u[k, m_next] - 2 * u[k, m] + u[k, m_prev]) + \
                       tau * f(k * tau, m * h)
        
        u[k+1, 0] = u[k+1, M]
    
    return u

def compute_error(numerical_sol, analytical_sol):
    return np.sqrt(np.mean((numerical_sol - analytical_sol) ** 2))

def run_convergence_analysis(scheme_func, scheme_name):
    grid_sizes = [20, 40, 80, 160, 320]
    errors = []
    
    for N in grid_sizes:
        K = N
        M = N
        
        u = scheme_func(K, M)
        
        x_values = np.linspace(0, X, M+1)
        u_analytical = analytical_solution(T, x_values)
        
        error = compute_error(u[-1, :], u_analytical)
        errors.append(error)
        
        print(f"{scheme_name} - Grid {N}x{N}: Error = {error:.6e}")
    
    convergence_orders = []
    for i in range(1, len(grid_sizes)):
        order = np.log(errors[i-1] / errors[i]) / np.log(grid_sizes[i] / grid_sizes[i-1])
        convergence_orders.append(order)
        print(f"Convergence order from {grid_sizes[i-1]} to {grid_sizes[i]}: {order:.3f}")
    
    return errors, convergence_orders

def plot_solution_comparison(K, M):
    u_corner = corner_scheme(K, M)
    u_rectangle = rectangle_scheme(K, M)
    u_laxwendroff = lax_wendroff_scheme(K, M)
    
    x_values = np.linspace(0, X, M+1)
    u_analytical = analytical_solution(T, x_values)
    
    plt.figure(figsize=(12, 8))
    plt.plot(x_values, u_analytical, 'k-', linewidth=2, label='Analytical')
    plt.plot(x_values, u_corner[-1, :], 'r-', label='Corner Scheme')
    plt.plot(x_values, u_rectangle[-1, :], 'g-', label='Rectangle Scheme')
    plt.plot(x_values, u_laxwendroff[-1, :], 'b-', label='Lax-Wendroff Scheme')
    
    plt.xlabel('x')
    plt.ylabel('u(T,x)')
    plt.title(f'Solution Comparison at T={T} (Grid {K}x{M})')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(f'solution_comparison_{M}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_convergence_comparison():
    errors_corner, orders_corner = run_convergence_analysis(corner_scheme, "Corner Scheme")
    errors_rectangle, orders_rectangle = run_convergence_analysis(rectangle_scheme, "Rectangle Scheme")
    errors_laxwendroff, orders_laxwendroff = run_convergence_analysis(lax_wendroff_scheme, "Lax-Wendroff Scheme")
    
    avg_order_corner = np.mean(orders_corner)
    avg_order_rectangle = np.mean(orders_rectangle)
    avg_order_laxwendroff = np.mean(orders_laxwendroff)
    
    print(f"===== FINAL CONVERGENCE ORDERS =====")
    print(f"Corner Scheme (Expected ~1.0): {avg_order_corner}")
    print(f"Rectangle Scheme (Expected ~1.0-2.0): {avg_order_rectangle}")
    print(f"Lax-Wendroff Scheme (Expected ~2.0): {avg_order_laxwendroff}")
    
    grid_sizes = [20, 40, 80, 160, 320]
    
    plt.figure(figsize=(12, 8))
    plt.loglog(grid_sizes, errors_corner, 'ro-', linewidth=2, 
               label=f'Corner Scheme (Expected Order: 1)')
    plt.loglog(grid_sizes, errors_rectangle, 'gs-', linewidth=2,
               label=f'Rectangle Scheme (Expected Order: 1-2)')
    plt.loglog(grid_sizes, errors_laxwendroff, 'b^-', linewidth=2,
               label=f'Lax-Wendroff Scheme (Expected Order: 2)')
    
    min_grid = grid_sizes[0]
    max_grid = grid_sizes[-1]
    ref_grids = np.logspace(np.log10(min_grid), np.log10(max_grid), 100)
    
    scale_1st = errors_corner[0] / (min_grid ** -1)
    scale_2nd = errors_laxwendroff[0] / (min_grid ** -2)
    
    plt.loglog(ref_grids, scale_1st * ref_grids ** -1, 'k--', label='1st Order')
    plt.loglog(ref_grids, scale_2nd * ref_grids ** -2, 'k:', label='2nd Order')
    
    plt.xlabel('Grid Size (N)')
    plt.ylabel('L2 Error')
    plt.title('Convergence Analysis - All Schemes (Fixed)')
    plt.grid(True)
    plt.legend()
    
    plt.savefig('convergence_all_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print(f"Initial condition: u(x,0) = sin^4(πx/5)")
    print(f"Domain length: X = {X}")
    print(f"Final time: T = {T}")
    
    print("\nGenerating solution comparison plot...")
    plot_solution_comparison(K=100, M=100)
    
    print("\nRunning convergence analysis for all schemes:")
    print("This may take a minute for the finer grids...\n")
    plot_convergence_comparison() 