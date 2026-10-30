"""Scoping regressions: capture-avoiding Function/With substitution, Module
closures (Temporary-symbol semantics) and complete Block save/restore.

Expected values follow Wolfram Language semantics.
"""
import pytest
import sympy as sp

from mikoshilang import Session
from mikoshilang.math_ops import to_sympy


@pytest.fixture
def s():
    return Session()


def ev(s, src):
    return repr(s.evaluate(src))


# ───────────── 1. Function: shadowing and capture avoidance ─────────────

@pytest.mark.parametrize("src,expected", [
    ("Function[x,Function[x,x]][1][2]", "2"),
    ("Function[x,Function[x,x]][1]", "Function[x, x]"),
    ("Function[{x},Function[{x},x]][1][2]", "2"),
    ("Function[x,Function[{y,x},x+y]][1][2,3]", "5"),
    ("Function[y,Function[x,x+y]][x]", "Function[x$, x$ + x]"),
    ("Function[y,Function[x,x+y]][x][5]", "5 + x"),
    ("Function[y,Function[x,x+y]][x$][5]", "5 + x$"),
    ("Function[y,Function[{x,z},x+y+z]][x+z][1,2]", "3 + x + z"),
    ("makeAdder[n_]:=Function[x,x+n]; makeAdder[x][1]", "1 + x"),
    ("makeAdder[n_]:=Function[x,x+n]; add2=makeAdder[2]; add2[10]", "12"),
    ("Function[x, Module[{x=3},x]][5]", "3"),
    ("Function[x, With[{x=3},x]][5]", "3"),
    ("x=1; Function[x, Block[{x=3},x]][5]", "3"),
    ("Function[y, Module[{x=1}, x+y]][x]", "1 + x"),
    ("Function[y, With[{x=1}, x+y]][x]", "1 + x"),
    ("Function[x, g[1] /. g[x_] :> x][5]", "1"),
    ("Function[x, g[1] /. g[x_] -> x][5]", "1"),
    ("Function[x, Cases[{1,-1,2}, y_ /; y > x]][0]", "{1, 2}"),
    ("Function[n, f[x_] := x + n][3]; f[1]", "4"),
    ("Function[n, f[n_] := n][3]; f[7]", "7"),
    ("Function[x, Hold[x]][5]", "Hold[5]"),
    ("Function[x, Function[#+x]][1][2]", "3"),
    ("f[x_] := Module[{y = x}, y]; f[y]", "y"),
    ("f[x_] := Function[y, x + y]; f[y][1]", "1 + y"),
])
def test_function_scoping(s, src, expected):
    assert ev(s, src) == expected


# ───────────── With ─────────────

@pytest.mark.parametrize("src,expected", [
    ("With[{x=1},Function[x,x]][2]", "2"),
    ("With[{x=1},Module[{x},x=7;x]]", "7"),
    ("With[{x=1}, With[{x=2}, x]]", "2"),
    ("With[{y=x},Function[x,x+y]][3]", "3 + x"),
    ("With[{x = y}, With[{y = 2}, x + y]]", "2 + y"),
    ("With[{x=2}, f[x_]:=x]; f[9]", "9"),
    ("With[{n=2}, f[x_]:=x^n]; f[3]", "9"),
    ("With[{x=1, y=2}, x+y]", "3"),
    ("x=5; With[{x=x+1}, x]", "6"),
])
def test_with_scoping(s, src, expected):
    assert ev(s, src) == expected


# ───────────── 2. Module closures ─────────────

@pytest.mark.parametrize("src,expected", [
    ("f=Module[{x=2},Function[y,x+y]]; f[3]", "5"),
    ("counter=Module[{c=0},Function[c=c+1]]; counter[]; counter[]", "2"),
    ("Module[{x=2}, g[y_]:=x+y]; g[1]", "3"),
    ("Module[{a}, Module[{x=2}, a = Function[x]; 0]; a[]]", "2"),
    ("h[] := Module[{x=4}, Return[Function[x]]]; h[][]", "4"),
    ("Catch[Module[{x=4}, Throw[Function[x+1]]]][]", "5"),
    ("Module[{x=2,y}, y:=x; Hold[y]] // ReleaseHold", "2"),
    ("adders = Table[Module[{k=i}, Function[z, z+k]], {i,3}]; Map[#[10]&, adders]", "{11, 12, 13}"),
    ("mk[n_] := Module[{acc=n}, {Function[acc=acc+1], Function[acc]}]; p=mk[10]; p[[1]][]; p[[1]][]; p[[2]][]",
     "12"),
    ("Module[{x=1},Module[{x=2},x]]", "2"),
])
def test_module_closures(s, src, expected):
    assert ev(s, src) == expected


def test_module_independent_closures(s):
    s.evaluate("mk[] := Module[{c=0}, Function[c=c+1]]; a=mk[]; b=mk[]; a[]; a[]; b[]")
    assert s.evaluate("{a[], b[]}") == s.evaluate("{3, 2}")


def test_module_unreferenced_locals_are_removed(s):
    s.evaluate("Do[Module[{x=i, f}, f[n_] := n + x; f[1]], {i, 50}]; Module[{t=1}, t+1]")
    assert s.state.own == {}
    assert not s.state.down and not s.state.exact
    assert s.state.temporaries == set()


def test_module_escaping_locals_survive_only_when_referenced(s):
    s.evaluate("c = Module[{k=0}, Function[k = k + 1]]; Module[{dead=1}, dead]")
    assert set(s.state.own) == {"c"} | s.state.temporaries
    assert len(s.state.temporaries) == 1
    assert s.evaluate("c[]") == 1


def test_module_cleans_up_after_error(s):
    with pytest.raises(Exception):
        s.evaluate("Module[{x=1}, Part[{1}, 5]]")
    assert s.state.own == {} and s.state.temporaries == set()


# ───────────── 3. Block: full save and restore ─────────────

@pytest.mark.parametrize("src,expected", [
    ("f[1]=10; Block[{f},f[1]=20]; f[1]", "10"),
    ("f[1]=10; Block[{f},f[1]]", "f[1]"),
    ("Block[{f},f[x_]:=x^2]; f[3]", "f[3]"),
    ("g[x_]:=x+1; Block[{g},g[x_]:=x^2; g[3]]", "9"),
    ("g[x_]:=x+1; Block[{g},g[x_]:=x^2]; g[3]", "4"),
    ("g[x_]:=x+1; Block[{g}, g[3]]", "g[3]"),
    ("g[x_]:=x+1; g[0]=100; Block[{g}, g[0]=-1; g[x_]:=0]; {g[0], g[5]}", "{100, 6}"),
    ("f[a][b]=3; Block[{f}, f[a][b]]", "f[a][b]"),
    ("f[a][b]=3; Block[{f}, f[a][b]=4]; f[a][b]", "3"),
    ("f[a][x_]:=x; Block[{f}, f[a][x_]:=-x]; f[a][2]", "2"),
    ("SetAttributes[f,Listable]; Block[{f}, Attributes[f]]", "{}"),
    ("SetAttributes[f,Listable]; Block[{f}, SetAttributes[f,HoldAll]]; Attributes[f]", "{Listable}"),
    ("Block[{f}, SetAttributes[f,Listable]]; Attributes[f]", "{}"),
    ("f[x_]:=x; Block[{f}, Clear[f]]; f[3]", "3"),
    ("x=1; Block[{x=2}, Block[{x=3}, x]; x]", "2"),
    ("x=1; Block[{x=2}, Block[{x=3}, x]]; x", "1"),
    ("x=1; Block[{x=2}, Block[{x}, x]]", "x"),
    ("x=7; Block[{x=x+1}, x]", "8"),
    ("x=1; h[]:=Block[{x=5},x]; {h[], x}", "{5, 1}"),
    ("x=1; g[]:=x; h[]:=Block[{x=5}, g[]]; {h[], g[]}", "{5, 1}"),
    ("fact[n_]:=Block[{k=n}, If[k<=1,1,k*fact[k-1]]]; fact[5]", "120"),
    ("x=1; Catch[Block[{x=2}, Throw[x]]]; x", "1"),
    ("f[1]=10; Catch[Block[{f}, f[1]=20; Throw[0]]]; f[1]", "10"),
    ("h[] := Block[{x=5}, Return[x]]; x=1; {h[], x}", "{5, 1}"),
    ("x=1; Do[Block[{x=i}, If[i==2, Break[]]], {i, 5}]; x", "1"),
    ("x := y + 1; Block[{x = 3}, x]", "3"),
    ("x := y + 1; Block[{x = 3}, x]; y = 1; x", "2"),
])
def test_block_restore(s, src, expected):
    assert ev(s, src) == expected


def test_block_restores_after_python_error(s):
    s.evaluate("f[1]=10; f[x_]:=x; x=4; SetAttributes[f, Listable]")
    before = _dump(s, "f"), _dump(s, "x")
    with pytest.raises(Exception):
        s.evaluate("Block[{f, x=9}, f[1]=0; f[x_]:=0; x=2; Part[{1}, 5]]")
    assert (_dump(s, "f"), _dump(s, "x")) == before
    assert s.evaluate("{f[1], f[2], x}") == s.evaluate("{10, 2, 4}")


def _dump(s, name):
    st = s.state
    return (repr(st.own.get(name, "<none>")),
            sorted(repr(v) for v in st.exact.values() if _key(v[0]) == name),
            [repr(d[:2]) for d in st.down.get(name, [])],
            sorted(st.attrs.get(name, ())))


def _key(lhs):
    from mikoshilang.kernel import _def_key
    return _def_key(lhs)
