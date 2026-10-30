"""Value-based regressions for the October 2026 audit (no external services)."""
import hashlib
import os
import subprocess
import sys
import pytest
import sympy as sp
from mikoshilang import Session, parse, ParseError, Expr, Symbol, N, Pi
from mikoshilang.math_ops import to_sympy
from mikoshilang.pattern import match
from mikoshilang.extended_helper import make_rule

@pytest.fixture
def session(): return Session()

@pytest.mark.parametrize('source', ['y’’[x]', '1+2 ]] junk', 'x % 2', 'x ~ y', 'x === ', '"unterminated', '(* unterminated', '1:2', 'x_.', 'f[[1]', '{1,2}.{3,4}'])
def test_malformed_input_rejected(source):
    with pytest.raises(ParseError): parse(source)

@pytest.mark.parametrize('source,expected', [
    ('f[x_] := x^2; f[3]',9),
    ('x^2 /. x -> 3',9),
    ('{1,2,3} /. x_Integer -> x^2',Expr('List',1,4,9)),
    ('Map[#^2 &, {1,2,3}]',Expr('List',1,4,9)),
    ('(#1 + #2 &)[3,4]',7),
    ('Apply[Plus,{1,2,3}]',6),
    ('Plus @@ {1,2,3}',6),
    ('f[x_] := x^2; f /@ {1,2,3}',Expr('List',1,4,9)),
    ('a=3\na+2',5),
    ('1 +\n2',3),
    ('Distance2D[{0,0},{3,4}]',5),
    ('Solve[x^2 - 4 == 0,x]',Expr('List',Expr('List',Expr('Rule',Symbol('x'),-2)),Expr('List',Expr('Rule',Symbol('x'),2)))),
    ('Solve[x^2 - 9 == 0,x]',Expr('List',Expr('List',Expr('Rule',Symbol('x'),-3)),Expr('List',Expr('Rule',Symbol('x'),3)))),
    ('Table[i^2,{i,1,5}]',Expr('List',1,4,9,16,25)),
    ('Range[5,1,-2]',Expr('List',5,3,1)),
    ('If[1 == 1,42,RiemannTensor[g,x]]',42),
    ('If[2 < 1,RiemannTensor[g,x],7]',7),
    ('f[x_Integer] := x+1; f[4]',5),
    ('f[x_,x_] := 9; f[2,2]',9),
    ('f[x__] := {x}; f[1,2,3]',Expr('List',1,2,3)),
    ('f[x___] := {x}; f[]',Expr('List')),
    ('x /. x :> 3',3),
    ('Dot[{{1,2},{3,4}},{{5,6},{7,8}}]',Expr('List',Expr('List',19,22),Expr('List',43,50))),
])
def test_language_values(session,source,expected):
    assert session.evaluate(source)==expected

@pytest.mark.parametrize('source,expected', [('x+x',2*sp.Symbol('x')),('2+x+3',sp.Symbol('x')+5),('1/3+1/3',sp.Rational(2,3)),('D[x^5,{x,2}]',20*sp.Symbol('x')**3)])
def test_exact_algebra(session,source,expected):
    assert to_sympy(session.evaluate(source)) == expected

def test_precision_not_machine_float(session):
    result=session.evaluate('N[Pi,50]')
    assert isinstance(result,sp.Float)
    assert abs(result-sp.pi.evalf(70)) < sp.Float('1e-49')
    assert abs(N(Pi,60)-sp.pi.evalf(70)) < sp.Float('1e-59')

def test_session_isolation():
    a,b=Session(),Session()
    a.evaluate('x=5; f[t_] := t+1')
    assert a.evaluate('x')==5
    assert b.evaluate('x')==Symbol('x')
    assert b.evaluate('f[3]')==Expr('f',3)
    a.clear()
    assert a.evaluate('x')==Symbol('x')

def test_reassignment_and_laziness(session):
    assert session.evaluate('x=2; y:=x+1; x=5; y')==6
    assert session.evaluate('x=2; y=x+1; x=5; y')==3

def test_replacement_is_single_pass(session):
    assert session.evaluate('x /. {x -> y, y -> 3}')==Symbol('y')

def test_parsed_patterns():
    assert match(parse('f[x_Integer]'),parse('f[3]'))=={'x':3}
    assert match(parse('f[x_,x_]'),parse('f[2,3]')) is None

def test_destructuring_adapter():
    rule=make_rule('Distance[{x_,y_},{a_,b_}]',lambda x,y,a,b:sp.sqrt((x-a)**2+(y-b)**2))
    assert rule.apply(parse('Distance[{0,0},{3,4}]'))==5

def test_sequence_adapter():
    rule=make_rule('SumValues[{xs__}]',lambda xs:sum(xs))
    assert rule.apply(parse('SumValues[{1,2,3}]'))==6

@pytest.mark.parametrize('pattern',['Broken Name[x_]','F[x_] junk','not a pattern'])
def test_bad_registry_entries_rejected(pattern):
    with pytest.raises((ValueError,ParseError)): make_rule(pattern,lambda x:x)

@pytest.mark.parametrize('source',['RiemannTensor[g,x]','Power1[2]','AFTModel[1,2,3]','HodgeDual[x]'])
def test_quarantined_no_fake_answer(session,source):
    with pytest.raises(NotImplementedError):session.evaluate(source)

def test_real_hash_not_placeholder(session):
    for value in ['abc','different message']:
        assert session.evaluate(f'SHA256["{value}"]')==hashlib.sha256(value.encode()).hexdigest()

@pytest.mark.parametrize('source',['Range[n]','Range[1,5,0]','Range[1.5]','Table[i,{i,1,n}]','Table[i,{i,1,200000}]'])
def test_bad_iteration_rejected(session,source):
    with pytest.raises(ValueError):session.evaluate(source)

def test_knowledge_opt_in(session):
    with pytest.raises(PermissionError):session.evaluate('EntitySearch["anything"]')

def test_import_and_arithmetic_do_not_write_home(tmp_path):
    env=dict(os.environ,HOME=str(tmp_path),PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable,'-c','import mikoshilang.knowledge_full; from mikoshilang import Session; assert Session().evaluate("2+3")==5'],env=env,check=True)
    assert not (tmp_path/'.mikoshilang').exists()

def test_three_factor_bridge():
    from mikoshilang.extended import _to_sympy
    x,y,z=map(Symbol,['x','y','z'])
    assert _to_sympy(Expr('Times',x,y,z))==sp.prod(sp.symbols('x y z'))

def test_float_not_guessed_exact():
    from mikoshilang.extended import _to_sympy
    assert isinstance(_to_sympy(0.3333333333333333),sp.Float)

def test_legacy_runner_checks_expected():
    from test_all_functions import test_function
    assert test_function('Plus','2,3',6)[0]=='error'
    assert test_function('Plus','2,3',5)[0]=='ok'
    assert test_function('Plus','2,3')[0]!='ok'

def test_real_graph_diameter(session):
    pytest.importorskip('networkx')
    assert session.evaluate('GraphDiameter[{{1,2},{2,3}}]') == 2
    assert session.evaluate('GraphDiameter[{{1,2},{2,3},{3,4}}]') == 3

def test_knowledge_mocked_opt_in(monkeypatch):
    from mikoshilang import knowledge_full_rules
    calls=[]
    def search(query):
        calls.append(query)
        return [{'id':'Q42','label':'fixture'}]
    monkeypatch.setattr(knowledge_full_rules,'EntitySearch',search)
    assert Session(allow_network=True).evaluate('EntitySearch["fixture"]')==[{'id':'Q42','label':'fixture'}]
    assert calls==['fixture']
    with pytest.raises(PermissionError): Session().evaluate('EntitySearch["fixture"]')

def test_cache_lazy_roundtrip(tmp_path,monkeypatch):
    from mikoshilang import knowledge_full as k
    db=tmp_path/'cache'/'knowledge.db'
    monkeypatch.setattr(k,'CACHE_DB',db)
    assert not db.exists()
    k._cache_set('test',{'answer':42},'fixture',60,'CC0')
    value=k._cache_get('test')
    assert value['data']=={'answer':42}
    assert value['source']=='fixture'
    assert value['license']=='CC0'
    assert k.CacheStats()['total_entries']==1
    k.CacheClear()
    assert k._cache_get('test') is None

def test_escaped_string():
    assert parse(r'"a\"b"')=='a"b'

def test_higher_derivatives_against_sympy(session):
    x=sp.Symbol('x')
    for degree in range(1,8):
        for order in range(4):
            assert to_sympy(session.evaluate(f'D[x^{degree},{{x,{order}}}]'))==sp.diff(x**degree,x,order)

def test_definition_values_against_sympy(session):
    x=sp.Symbol('x')
    session.evaluate('poly[x_] := x^3 - 2*x + 1')
    for value in range(-10,11):
        assert session.evaluate(f'poly[{value}]')==(x**3-2*x+1).subs(x,value)

def test_function_value_assignment(session):
    assert session.evaluate('square = #^2 &; square[5]')==25
    assert session.evaluate('Map[square,{1,2,3}]')==Expr('List',1,4,9)

def test_short_circuit_logic(session):
    assert session.evaluate('False && RiemannTensor[g,x]') is False
    assert session.evaluate('True || RiemannTensor[g,x]') is True
    assert session.evaluate('If[True && !False,8,9]')==8

def test_boolean_not_integer_pattern():
    assert match(parse('f[x_Integer]'),parse('f[True]')) is None

@pytest.mark.parametrize('source',['Simplify["x+1"]','N["2+2"]','Solve["x^2-4",x]'])
def test_no_implicit_sympy_string_evaluation(session,source):
    with pytest.raises(TypeError): session.evaluate(source)

def test_symbolic_numerical_coefficients(session):
    result=to_sympy(session.evaluate('N[Pi+x,50]'))
    constant=result.subs(sp.Symbol('x'),0)
    assert abs(constant-sp.pi.evalf(70)) < sp.Float('1e-49')
