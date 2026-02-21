"""Tests for advanced visualization module."""

import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

from mikoshilang.expr import Expr, Symbol
from mikoshilang.parser import parse
from mikoshilang.visualization import (
    Plot2D, Plot3D, ContourPlot, VectorFieldPlot,
    ParametricPlot, PolarPlot, Heatmap
)


class TestVisualization:
    """Test suite for visualization functions."""
    
    def test_plot2d_single(self):
        """Test 2D plotting of single function."""
        x = Symbol('x')
        expr = Expr("Power", x, 2)
        var_range = Expr("List", x, -5, 5)
        
        fig = Plot2D(expr, var_range, show=False)
        assert fig is not None
    
    def test_plot2d_multiple(self):
        """Test 2D plotting of multiple functions."""
        x = Symbol('x')
        funcs = Expr("List", 
                    Expr("Sin", x),
                    Expr("Cos", x))
        var_range = Expr("List", x, 0, 6.28)
        
        fig = Plot2D(funcs, var_range, show=False)
        assert fig is not None
    
    @pytest.mark.skip(reason="3D plotting requires specific matplotlib version")
    def test_plot3d(self):
        """Test 3D surface plot."""
        x = Symbol('x')
        y = Symbol('y')
        expr = Expr("Plus", 
                   Expr("Power", x, 2),
                   Expr("Power", y, 2))
        x_range = Expr("List", x, -3, 3)
        y_range = Expr("List", y, -3, 3)
        
        fig = Plot3D(expr, x_range, y_range, show=False)
        assert fig is not None
    
    def test_contour_plot(self):
        """Test contour plot."""
        x = Symbol('x')
        y = Symbol('y')
        expr = Expr("Sin", Expr("Times", x, y))
        x_range = Expr("List", x, -3, 3)
        y_range = Expr("List", y, -3, 3)
        
        fig = ContourPlot(expr, x_range, y_range, show=False)
        assert fig is not None
    
    def test_vector_field(self):
        """Test vector field plot."""
        x = Symbol('x')
        y = Symbol('y')
        funcs = Expr("List", 
                    Expr("Times", -1, y),
                    x)
        x_range = Expr("List", x, -2, 2)
        y_range = Expr("List", y, -2, 2)
        
        fig = VectorFieldPlot(funcs, x_range, y_range, show=False)
        assert fig is not None
    
    def test_parametric_plot_2d(self):
        """Test 2D parametric plot."""
        t = Symbol('t')
        funcs = Expr("List",
                    Expr("Cos", t),
                    Expr("Sin", t))
        t_range = Expr("List", t, 0, 6.28)
        
        fig = ParametricPlot(funcs, t_range, show=False)
        assert fig is not None
    
    @pytest.mark.skip(reason="3D plotting requires specific matplotlib version")
    def test_parametric_plot_3d(self):
        """Test 3D parametric plot."""
        t = Symbol('t')
        funcs = Expr("List",
                    Expr("Cos", t),
                    Expr("Sin", t),
                    t)
        t_range = Expr("List", t, 0, 12.56)
        
        fig = ParametricPlot(funcs, t_range, show=False)
        assert fig is not None
    
    def test_polar_plot(self):
        """Test polar plot."""
        theta = Symbol('theta')
        expr = Expr("Plus", 1, Expr("Cos", theta))
        theta_range = Expr("List", theta, 0, 6.28)
        
        fig = PolarPlot(expr, theta_range, show=False)
        assert fig is not None
    
    def test_heatmap(self):
        """Test heatmap visualization."""
        data = Expr("List",
                   Expr("List", 1, 2, 3),
                   Expr("List", 4, 5, 6),
                   Expr("List", 7, 8, 9))
        
        fig = Heatmap(data, show=False)
        assert fig is not None
