"""Advanced visualization — 3D plots, animations, interactive visualizations, and geographic maps.

Supports static (matplotlib), interactive (plotly), and animated plots.
"""

from __future__ import annotations
from typing import Any, Callable, List, Tuple, Optional
from .expr import Expr, Symbol
from .extended import _to_sympy, _extract_list, _to_float_list
import math


def _ensure_matplotlib():
    """Ensure matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("Matplotlib required. Install with: pip install matplotlib")


def _ensure_plotly():
    """Ensure plotly is available for interactive plots."""
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        raise ImportError("Plotly required for interactive plots. Install with: pip install plotly")


def _ensure_numpy():
    """Ensure numpy is available."""
    try:
        import numpy as np
        return np
    except ImportError:
        raise ImportError("NumPy required. Install with: pip install numpy")


# ── 2D Plotting ──

def Plot2D(expr, var_range, **kwargs):
    """
    2D plot with matplotlib.
    
    Args:
        expr: Expression to plot (or list of expressions)
        var_range: Expr("List", var, min, max) e.g. {x, -5, 5}
        **kwargs: style, color, linewidth, title, xlabel, ylabel, legend, grid
    
    Returns:
        matplotlib figure
    """
    plt = _ensure_matplotlib()
    np = _ensure_numpy()
    import sympy as sp
    
    # Parse range
    range_list = _extract_list(var_range)
    if len(range_list) != 3:
        raise ValueError("Range must be {var, min, max}")
    var, x_min, x_max = range_list
    
    # Handle multiple expressions
    if isinstance(expr, Expr) and expr.head == "List":
        expressions = _extract_list(expr)
    else:
        expressions = [expr]
    
    # Convert to sympy and create lambdas
    sym_var = _to_sympy(var)
    functions = []
    for e in expressions:
        sym_expr = _to_sympy(e)
        f = sp.lambdify(sym_var, sym_expr, modules=['numpy'])
        functions.append((f, str(e)))
    
    # Generate points
    x_vals = np.linspace(float(x_min), float(x_max), kwargs.get('points', 500))
    
    # Create plot
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    
    colors = kwargs.get('colors', None)
    if not colors:
        colors = plt.cm.tab10.colors
    
    for i, (f, label) in enumerate(functions):
        try:
            y_vals = f(x_vals)
            color = colors[i % len(colors)] if isinstance(colors, (list, tuple)) else kwargs.get('color')
            ax.plot(x_vals, y_vals, 
                   label=label if len(functions) > 1 else None,
                   color=color,
                   linewidth=kwargs.get('linewidth', 2),
                   linestyle=kwargs.get('linestyle', '-'))
        except Exception as e:
            print(f"Warning: Could not plot {label}: {e}")
    
    # Styling
    ax.set_xlabel(kwargs.get('xlabel', str(var)), fontsize=12)
    ax.set_ylabel(kwargs.get('ylabel', 'f(x)'), fontsize=12)
    ax.set_title(kwargs.get('title', f'Plot of {expressions[0]}'), fontsize=14)
    ax.grid(kwargs.get('grid', True), alpha=0.3)
    
    if len(functions) > 1 and kwargs.get('legend', True):
        ax.legend()
    
    if kwargs.get('show', True):
        plt.show()
    
    return fig


def Plot3D(expr, x_range, y_range, **kwargs):
    """
    3D surface plot.
    
    Args:
        expr: Expression f(x, y)
        x_range: {x, x_min, x_max}
        y_range: {y, y_min, y_max}
        **kwargs: points, cmap, alpha, title, interactive (bool)
    
    Returns:
        matplotlib figure or plotly figure if interactive=True
    """
    np = _ensure_numpy()
    import sympy as sp
    
    # Parse ranges
    x_list = _extract_list(x_range)
    y_list = _extract_list(y_range)
    x_var, x_min, x_max = x_list
    y_var, y_min, y_max = y_list
    
    # Convert to sympy
    sym_expr = _to_sympy(expr)
    sym_x = _to_sympy(x_var)
    sym_y = _to_sympy(y_var)
    f = sp.lambdify((sym_x, sym_y), sym_expr, modules=['numpy'])
    
    # Generate mesh
    points = kwargs.get('points', 100)
    x_vals = np.linspace(float(x_min), float(x_max), points)
    y_vals = np.linspace(float(y_min), float(y_max), points)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    try:
        Z = f(X, Y)
    except Exception as e:
        raise ValueError(f"Could not evaluate function: {e}")
    
    # Interactive plotly version
    if kwargs.get('interactive', False):
        go = _ensure_plotly()
        
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, 
                                         colorscale=kwargs.get('colorscale', 'Viridis'))])
        fig.update_layout(
            title=kwargs.get('title', f'3D Plot of {expr}'),
            scene=dict(
                xaxis_title=str(x_var),
                yaxis_title=str(y_var),
                zaxis_title='f(x,y)'
            ),
            width=kwargs.get('width', 800),
            height=kwargs.get('height', 600)
        )
        
        if kwargs.get('show', True):
            fig.show()
        return fig
    
    # Static matplotlib version
    plt = _ensure_matplotlib()
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=kwargs.get('figsize', (10, 8)))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, 
                          cmap=kwargs.get('cmap', 'viridis'),
                          alpha=kwargs.get('alpha', 0.9),
                          edgecolor='none')
    
    ax.set_xlabel(str(x_var), fontsize=12)
    ax.set_ylabel(str(y_var), fontsize=12)
    ax.set_zlabel('f(x,y)', fontsize=12)
    ax.set_title(kwargs.get('title', f'3D Plot of {expr}'), fontsize=14)
    
    fig.colorbar(surf, ax=ax, shrink=0.5)
    
    if kwargs.get('show', True):
        plt.show()
    
    return fig


def ContourPlot(expr, x_range, y_range, **kwargs):
    """
    Contour plot of f(x, y).
    
    Args:
        expr: Expression f(x, y)
        x_range: {x, x_min, x_max}
        y_range: {y, y_min, y_max}
        **kwargs: levels, filled, cmap, contour_labels
    """
    plt = _ensure_matplotlib()
    np = _ensure_numpy()
    import sympy as sp
    
    # Parse ranges
    x_list = _extract_list(x_range)
    y_list = _extract_list(y_range)
    x_var, x_min, x_max = x_list
    y_var, y_min, y_max = y_list
    
    # Convert to sympy
    sym_expr = _to_sympy(expr)
    sym_x = _to_sympy(x_var)
    sym_y = _to_sympy(y_var)
    f = sp.lambdify((sym_x, sym_y), sym_expr, modules=['numpy'])
    
    # Generate mesh
    points = kwargs.get('points', 200)
    x_vals = np.linspace(float(x_min), float(x_max), points)
    y_vals = np.linspace(float(y_min), float(y_max), points)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f(X, Y)
    
    # Create plot
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))
    
    levels = kwargs.get('levels', 20)
    
    if kwargs.get('filled', True):
        contour = ax.contourf(X, Y, Z, levels=levels, cmap=kwargs.get('cmap', 'viridis'))
        fig.colorbar(contour, ax=ax)
    else:
        contour = ax.contour(X, Y, Z, levels=levels, cmap=kwargs.get('cmap', 'viridis'))
        if kwargs.get('contour_labels', True):
            ax.clabel(contour, inline=True, fontsize=8)
    
    ax.set_xlabel(str(x_var), fontsize=12)
    ax.set_ylabel(str(y_var), fontsize=12)
    ax.set_title(kwargs.get('title', f'Contour Plot of {expr}'), fontsize=14)
    ax.grid(kwargs.get('grid', True), alpha=0.3)
    
    if kwargs.get('show', True):
        plt.show()
    
    return fig


def VectorFieldPlot(funcs, x_range, y_range, **kwargs):
    """
    Vector field plot.
    
    Args:
        funcs: {Fx, Fy} — vector components
        x_range: {x, x_min, x_max}
        y_range: {y, y_min, y_max}
        **kwargs: density, color, normalize, streamlines
    """
    plt = _ensure_matplotlib()
    np = _ensure_numpy()
    import sympy as sp
    
    # Parse
    func_list = _extract_list(funcs)
    if len(func_list) != 2:
        raise ValueError("Need 2 components: {Fx, Fy}")
    
    x_list = _extract_list(x_range)
    y_list = _extract_list(y_range)
    x_var, x_min, x_max = x_list
    y_var, y_min, y_max = y_list
    
    # Convert to sympy
    sym_x = _to_sympy(x_var)
    sym_y = _to_sympy(y_var)
    Fx = sp.lambdify((sym_x, sym_y), _to_sympy(func_list[0]), modules=['numpy'])
    Fy = sp.lambdify((sym_x, sym_y), _to_sympy(func_list[1]), modules=['numpy'])
    
    # Generate grid
    density = kwargs.get('density', 20)
    x_vals = np.linspace(float(x_min), float(x_max), density)
    y_vals = np.linspace(float(y_min), float(y_max), density)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    U = Fx(X, Y)
    V = Fy(X, Y)
    
    # Normalize if requested
    if kwargs.get('normalize', False):
        M = np.sqrt(U**2 + V**2)
        M[M == 0] = 1  # Avoid division by zero
        U, V = U/M, V/M
    
    # Create plot
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))
    
    if kwargs.get('streamlines', False):
        ax.streamplot(X, Y, U, V, 
                     color=kwargs.get('color', 'blue'),
                     density=kwargs.get('stream_density', 1.5),
                     linewidth=kwargs.get('linewidth', 1),
                     arrowsize=kwargs.get('arrowsize', 1.5))
    else:
        ax.quiver(X, Y, U, V, 
                 color=kwargs.get('color', 'blue'),
                 alpha=kwargs.get('alpha', 0.8),
                 scale=kwargs.get('scale', None))
    
    ax.set_xlabel(str(x_var), fontsize=12)
    ax.set_ylabel(str(y_var), fontsize=12)
    ax.set_title(kwargs.get('title', 'Vector Field'), fontsize=14)
    ax.grid(kwargs.get('grid', True), alpha=0.3)
    
    if kwargs.get('show', True):
        plt.show()
    
    return fig


def ParametricPlot(funcs, t_range, **kwargs):
    """
    Parametric plot (x(t), y(t)) or 3D (x(t), y(t), z(t)).
    
    Args:
        funcs: {x(t), y(t)} or {x(t), y(t), z(t)}
        t_range: {t, t_min, t_max}
        **kwargs: color, linewidth, animate
    """
    plt = _ensure_matplotlib()
    np = _ensure_numpy()
    import sympy as sp
    
    # Parse
    func_list = _extract_list(funcs)
    t_list = _extract_list(t_range)
    t_var, t_min, t_max = t_list
    
    # Convert to sympy
    sym_t = _to_sympy(t_var)
    lambdas = [sp.lambdify(sym_t, _to_sympy(f), modules=['numpy']) for f in func_list]
    
    # Generate parameter values
    t_vals = np.linspace(float(t_min), float(t_max), kwargs.get('points', 500))
    coords = [f(t_vals) for f in lambdas]
    
    # 2D or 3D?
    if len(coords) == 2:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))
        ax.plot(coords[0], coords[1], 
               color=kwargs.get('color', 'blue'),
               linewidth=kwargs.get('linewidth', 2))
        ax.set_xlabel('x(t)', fontsize=12)
        ax.set_ylabel('y(t)', fontsize=12)
        ax.grid(kwargs.get('grid', True), alpha=0.3)
    
    elif len(coords) == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=kwargs.get('figsize', (10, 8)))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(coords[0], coords[1], coords[2],
               color=kwargs.get('color', 'blue'),
               linewidth=kwargs.get('linewidth', 2))
        ax.set_xlabel('x(t)', fontsize=12)
        ax.set_ylabel('y(t)', fontsize=12)
        ax.set_zlabel('z(t)', fontsize=12)
    
    else:
        raise ValueError("Need 2 or 3 parametric functions")
    
    ax.set_title(kwargs.get('title', 'Parametric Plot'), fontsize=14)
    
    if kwargs.get('show', True):
        plt.show()
    
    return fig


def PolarPlot(r_expr, theta_range, **kwargs):
    """
    Polar plot r(θ).
    
    Args:
        r_expr: r as function of theta
        theta_range: {theta, theta_min, theta_max}
    """
    plt = _ensure_matplotlib()
    np = _ensure_numpy()
    import sympy as sp
    
    # Parse
    theta_list = _extract_list(theta_range)
    theta_var, theta_min, theta_max = theta_list
    
    # Convert to sympy
    sym_theta = _to_sympy(theta_var)
    sym_r = _to_sympy(r_expr)
    r_func = sp.lambdify(sym_theta, sym_r, modules=['numpy'])
    
    # Generate values
    theta_vals = np.linspace(float(theta_min), float(theta_max), kwargs.get('points', 500))
    r_vals = r_func(theta_vals)
    
    # Create polar plot
    fig = plt.figure(figsize=kwargs.get('figsize', (8, 8)))
    ax = fig.add_subplot(111, projection='polar')
    
    ax.plot(theta_vals, r_vals, 
           color=kwargs.get('color', 'blue'),
           linewidth=kwargs.get('linewidth', 2))
    ax.set_title(kwargs.get('title', f'Polar Plot of r = {r_expr}'), 
                fontsize=14, pad=20)
    ax.grid(kwargs.get('grid', True), alpha=0.3)
    
    if kwargs.get('show', True):
        plt.show()
    
    return fig


# ── Animations ──

def AnimatePlot(expr, var, param_range, **kwargs):
    """
    Animate a plot as a parameter changes.
    
    Args:
        expr: Expression depending on var and a parameter
        var: Independent variable
        param_range: {param, min, max, steps}
        **kwargs: x_range, y_range, interval (ms), save_as
    
    Returns:
        matplotlib animation
    """
    plt = _ensure_matplotlib()
    np = _ensure_numpy()
    import sympy as sp
    from matplotlib.animation import FuncAnimation
    
    # Parse parameter range
    param_list = _extract_list(param_range)
    if len(param_list) == 4:
        param, p_min, p_max, steps = param_list
    else:
        param, p_min, p_max = param_list
        steps = 50
    
    # Get x range
    x_range = kwargs.get('x_range', Expr("List", var, -10, 10))
    x_list = _extract_list(x_range)
    x_min, x_max = float(x_list[1]), float(x_list[2])
    
    # Convert to sympy
    sym_var = _to_sympy(var)
    sym_param = _to_sympy(param)
    sym_expr = _to_sympy(expr)
    
    # Create lambdified function
    f = sp.lambdify((sym_var, sym_param), sym_expr, modules=['numpy'])
    
    # Setup plot
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    x_vals = np.linspace(x_min, x_max, kwargs.get('points', 500))
    
    # Initial plot
    p_vals = np.linspace(float(p_min), float(p_max), int(steps))
    line, = ax.plot(x_vals, f(x_vals, p_vals[0]), 
                   color=kwargs.get('color', 'blue'),
                   linewidth=2)
    
    ax.set_xlabel(str(var), fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set y limits if provided
    if 'y_range' in kwargs:
        y_list = _extract_list(kwargs['y_range'])
        ax.set_ylim(float(y_list[0]), float(y_list[1]))
    
    title_template = kwargs.get('title', f'{param} = {{:.2f}}')
    title = ax.set_title(title_template.format(p_vals[0]), fontsize=14)
    
    def update(frame):
        p_val = p_vals[frame]
        y_vals = f(x_vals, p_val)
        line.set_ydata(y_vals)
        title.set_text(title_template.format(p_val))
        return line, title
    
    anim = FuncAnimation(fig, update, frames=len(p_vals),
                        interval=kwargs.get('interval', 50),
                        blit=True, repeat=kwargs.get('repeat', True))
    
    # Save if requested
    if 'save_as' in kwargs:
        anim.save(kwargs['save_as'], writer='pillow', fps=kwargs.get('fps', 20))
    
    if kwargs.get('show', True):
        plt.show()
    
    return anim


# ── Heatmaps & Matrix Visualization ──

def Heatmap(data, **kwargs):
    """
    Heatmap visualization of matrix data.
    
    Args:
        data: Matrix as nested lists or Expr
        **kwargs: cmap, annot (annotate), fmt, cbar (colorbar)
    """
    plt = _ensure_matplotlib()
    np = _ensure_numpy()
    
    # Extract data
    if isinstance(data, Expr) and data.head == "List":
        matrix = [_to_float_list(row) for row in _extract_list(data)]
    else:
        matrix = data
    
    matrix = np.array(matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))
    
    im = ax.imshow(matrix, cmap=kwargs.get('cmap', 'viridis'), 
                   aspect=kwargs.get('aspect', 'auto'))
    
    # Annotations
    if kwargs.get('annot', False):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = ax.text(j, i, f"{matrix[i, j]:{kwargs.get('fmt', '.2f')}}",
                             ha="center", va="center", color="white", fontsize=8)
    
    # Colorbar
    if kwargs.get('cbar', True):
        fig.colorbar(im, ax=ax)
    
    ax.set_title(kwargs.get('title', 'Heatmap'), fontsize=14)
    
    if kwargs.get('show', True):
        plt.show()
    
    return fig


# ── Interactive 3D ──

def Interactive3D(expr, x_range, y_range, **kwargs):
    """
    Interactive 3D plot using plotly.
    
    Args:
        expr: f(x, y)
        x_range: {x, x_min, x_max}
        y_range: {y, y_min, y_max}
        **kwargs: plot_type ('surface', 'wireframe', 'scatter3d')
    """
    return Plot3D(expr, x_range, y_range, interactive=True, **kwargs)


# Build visualization rules for integration into evaluate engine

def build_visualization_rules():
    """Build rules for visualization functions."""
    from .rules import RuleDelayed
    from .pattern import Blank
    
    rules = []
    
    def rule1(head, fn):
        rules.append(RuleDelayed(Expr(head, Blank("a")), lambda b: fn(b["a"])))
    
    def rule2(head, fn):
        rules.append(RuleDelayed(Expr(head, Blank("a"), Blank("b")), lambda b: fn(b["a"], b["b"])))
    
    def rule3(head, fn):
        rules.append(RuleDelayed(Expr(head, Blank("a"), Blank("b"), Blank("c")), lambda b: fn(b["a"], b["b"], b["c"])))
    
    # Register visualization functions
    rule2("Plot2D", Plot2D)
    rule3("Plot3D", Plot3D)
    rule3("ContourPlot", ContourPlot)
    rule3("VectorFieldPlot", VectorFieldPlot)
    rule2("ParametricPlot", ParametricPlot)
    rule2("PolarPlot", PolarPlot)
    rule3("AnimatePlot", AnimatePlot)
    rule1("Heatmap", Heatmap)
    rule3("Interactive3D", Interactive3D)
    
    return rules


VISUALIZATION_RULES = build_visualization_rules()
