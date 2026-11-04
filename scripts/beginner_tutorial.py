from fractions import Fraction
from mikoshilang import Session, List, Take, Range
s = Session()
assert s.eval('2 + 3 * 4') == 14
assert s.eval('1/3 + 1/6') == Fraction(1, 2)
assert s.eval('{2, 4, 6}') == [2, 4, 6]
s.eval('square[x_] := x^2')
assert s.eval('square[5]') == 25
assert s.eval('Table[square[k], {k, 1, 4}]') == [1, 4, 9, 16]
s.eval('cost[n_] := 12 + 3*n')
costs = s.eval('Table[cost[n], {n, 0, 5}]')
assert costs == [12, 15, 18, 21, 24, 27]
print(costs)
assert s.eval('cost[5]') == 27
assert Take(List(10, 20, 30, 40), -2).args == (30, 40)
assert Range(3, 1, -1).args == (3, 2, 1)
print('Beginner tutorial: all checks passed')
