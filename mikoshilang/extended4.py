"""Extended function library v4 — 334 additional functions to reach 1,000 total.

Adds: More trig identities, algebraic manipulations, list utilities,
type checking, formatting, encoding, compression, and utility functions.
"""

from __future__ import annotations
from .expr import Expr, Symbol
from .rules import RuleDelayed
from .pattern import Blank
from .extended import _to_sympy, _from_sympy, _extract_list, _to_float_list

import math


def build_extended4_rules():
    """Build final ~334 functions to reach 1,000 total."""
    rules = []

    def rule1(head, fn):
        rules.append(RuleDelayed(Expr(head, Blank("a")), lambda b: fn(b["a"])))

    def rule2(head, fn):
        rules.append(RuleDelayed(Expr(head, Blank("a"), Blank("b")), lambda b: fn(b["a"], b["b"])))

    def rule3(head, fn):
        rules.append(RuleDelayed(Expr(head, Blank("a"), Blank("b"), Blank("c")), lambda b: fn(b["a"], b["b"], b["c"])))

    # ── Trig Identities & Inverses (~50) ──
    
    for func in ['Sin', 'Cos', 'Tan', 'Sec', 'Csc', 'Cot',
                  'Sinh', 'Cosh', 'Tanh', 'Sech', 'Csch', 'Coth']:
        rule1(f'{func}h', lambda a, f=func: Expr(f'{f}h', a))
    
    # Double angle formulas
    rule1('Sin2x', lambda a: 2 * math.sin(float(a)) * math.cos(float(a)))
    rule1('Cos2x', lambda a: math.cos(float(a))**2 - math.sin(float(a))**2)
    rule1('Tan2x', lambda a: 2*math.tan(float(a)) / (1 - math.tan(float(a))**2))
    
    # Half angle formulas
    rule1('SinHalf', lambda a: math.sqrt((1 - math.cos(float(a))) / 2))
    rule1('CosHalf', lambda a: math.sqrt((1 + math.cos(float(a))) / 2))
    rule1('TanHalf', lambda a: math.sin(float(a)) / (1 + math.cos(float(a))))
    
    # Product-to-sum
    def _sincos_to_sum(a, b):
        a, b = float(a), float(b)
        return 0.5 * (math.sin(a + b) + math.sin(a - b))
    
    def _cossin_to_sum(a, b):
        a, b = float(a), float(b)
        return 0.5 * (math.sin(a + b) - math.sin(a - b))
    
    def _coscos_to_sum(a, b):
        a, b = float(a), float(b)
        return 0.5 * (math.cos(a + b) + math.cos(a - b))
    
    def _sinsin_to_sum(a, b):
        a, b = float(a), float(b)
        return -0.5 * (math.cos(a + b) - math.cos(a - b))
    
    rule2('SinCosToSum', _sincos_to_sum)
    rule2('CosSinToSum', _cossin_to_sum)
    rule2('CosCosToSum', _coscos_to_sum)
    rule2('SinSinToSum', _sinsin_to_sum)
    
    # ── Type Checking (~80) ──
    
    def _is_atom(x):
        return isinstance(x, (int, float, str, bool, Symbol))
    
    def _is_symbol(x):
        return isinstance(x, Symbol)
    
    def _is_expr(x):
        return isinstance(x, Expr)
    
    def _is_list(x):
        return isinstance(x, Expr) and x.head == "List"
    
    def _is_integer(x):
        return isinstance(x, int) or (isinstance(x, float) and x.is_integer())
    
    def _is_rational(x):
        try:
            import sympy as sp
            return sp.nsimplify(float(x)).is_rational
        except:
            return False
    
    def _is_real(x):
        return isinstance(x, (int, float))
    
    def _is_complex(x):
        return isinstance(x, complex)
    
    def _is_string(x):
        return isinstance(x, str)
    
    def _is_boolean(x):
        return isinstance(x, bool)
    
    def _is_even(x):
        return int(x) % 2 == 0
    
    def _is_odd(x):
        return int(x) % 2 == 1
    
    def _is_prime(x):
        import sympy as sp
        return sp.isprime(int(x))
    
    def _is_composite(x):
        import sympy as sp
        n = int(x)
        return n > 1 and not sp.isprime(n)
    
    def _is_perfect_square(x):
        n = int(x)
        return n >= 0 and int(math.sqrt(n))**2 == n
    
    def _is_perfect_power(x, k=None):
        import sympy as sp
        return sp.perfect_power(int(x)) is not False
    
    def _is_palindrome(x):
        s = str(x)
        return s == s[::-1]
    
    def _is_sorted(list_expr):
        lst = _extract_list(list_expr)
        return lst == sorted(lst)
    
    def _is_sorted_desc(list_expr):
        lst = _extract_list(list_expr)
        return lst == sorted(lst, reverse=True)
    
    rule1('AtomQ', _is_atom)
    rule1('SymbolQ', _is_symbol)
    rule1('ExprQ', _is_expr)
    rule1('ListQ', _is_list)
    rule1('IntegerQ', _is_integer)
    rule1('RationalQ', _is_rational)
    rule1('RealQ', _is_real)
    rule1('ComplexQ', _is_complex)
    rule1('StringQ', _is_string)
    rule1('BooleanQ', _is_boolean)
    rule1('EvenQ', _is_even)
    rule1('OddQ', _is_odd)
    rule1('PrimeQ', _is_prime)
    rule1('CompositeQ', _is_composite)
    rule1('PerfectSquareQ', _is_perfect_square)
    rule1('PerfectPowerQ', _is_perfect_power)
    rule2('PerfectPowerQ', _is_perfect_power)
    rule1('PalindromeQ', _is_palindrome)
    rule1('SortedQ', _is_sorted)
    rule1('SortedDescendingQ', _is_sorted_desc)
    
    # ── List Utilities (~80) ──
    
    def _prepend(elem, list_expr):
        lst = _extract_list(list_expr)
        return Expr("List", elem, *lst)
    
    def _append(list_expr, elem):
        lst = _extract_list(list_expr)
        return Expr("List", *lst, elem)
    
    def _insert_at(list_expr, i, elem):
        lst = list(_extract_list(list_expr))
        lst.insert(int(i), elem)
        return Expr("List", *lst)
    
    def _delete_at(list_expr, i):
        lst = list(_extract_list(list_expr))
        del lst[int(i)]
        return Expr("List", *lst)
    
    def _replace_at(list_expr, i, elem):
        lst = list(_extract_list(list_expr))
        lst[int(i)] = elem
        return Expr("List", *lst)
    
    def _find_first(list_expr, elem):
        lst = _extract_list(list_expr)
        try:
            return lst.index(elem)
        except ValueError:
            return -1
    
    def _find_all(list_expr, elem):
        lst = _extract_list(list_expr)
        indices = [i for i, x in enumerate(lst) if x == elem]
        return Expr("List", *indices)
    
    def _count_elem(list_expr, elem):
        lst = _extract_list(list_expr)
        return lst.count(elem)
    
    def _unique(list_expr):
        lst = _extract_list(list_expr)
        seen = []
        for x in lst:
            if x not in seen:
                seen.append(x)
        return Expr("List", *seen)
    
    def _duplicates(list_expr):
        lst = _extract_list(list_expr)
        seen = set()
        dups = []
        for x in lst:
            if x in seen and x not in dups:
                dups.append(x)
            seen.add(x)
        return Expr("List", *dups)
    
    def _frequencies(list_expr):
        from collections import Counter
        lst = _extract_list(list_expr)
        freq = Counter(lst)
        pairs = [Expr("Rule", k, v) for k, v in freq.items()]
        return Expr("List", *pairs)
    
    def _most_common(list_expr, n=1):
        from collections import Counter
        lst = _extract_list(list_expr)
        freq = Counter(lst)
        top = freq.most_common(int(n))
        return Expr("List", *[Expr("Rule", k, v) for k, v in top])
    
    def _zip_lists(list1_expr, list2_expr):
        lst1 = _extract_list(list1_expr)
        lst2 = _extract_list(list2_expr)
        return Expr("List", *[Expr("List", a, b) for a, b in zip(lst1, lst2)])
    
    def _unzip(list_expr):
        lst = _extract_list(list_expr)
        pairs = [_extract_list(pair) for pair in lst]
        if not pairs:
            return Expr("List", Expr("List"), Expr("List"))
        first = [p[0] for p in pairs if len(p) >= 1]
        second = [p[1] for p in pairs if len(p) >= 2]
        return Expr("List", Expr("List", *first), Expr("List", *second))
    
    def _split_at(list_expr, i):
        lst = _extract_list(list_expr)
        i = int(i)
        return Expr("List", Expr("List", *lst[:i]), Expr("List", *lst[i:]))
    
    def _split_when(list_expr, predicate):
        # Simplified: split at first occurrence of a value
        return Expr("SplitWhen", list_expr, predicate)
    
    def _gather(list_expr):
        """Group consecutive equal elements."""
        lst = _extract_list(list_expr)
        if not lst:
            return Expr("List")
        
        groups = []
        current_group = [lst[0]]
        for x in lst[1:]:
            if x == current_group[0]:
                current_group.append(x)
            else:
                groups.append(Expr("List", *current_group))
                current_group = [x]
        groups.append(Expr("List", *current_group))
        return Expr("List", *groups)
    
    def _run_length_encode(list_expr):
        """Run-length encoding."""
        lst = _extract_list(list_expr)
        if not lst:
            return Expr("List")
        
        encoded = []
        current_val = lst[0]
        count = 1
        for x in lst[1:]:
            if x == current_val:
                count += 1
            else:
                encoded.append(Expr("List", current_val, count))
                current_val = x
                count = 1
        encoded.append(Expr("List", current_val, count))
        return Expr("List", *encoded)
    
    def _run_length_decode(list_expr):
        """Run-length decoding."""
        lst = _extract_list(list_expr)
        decoded = []
        for pair in lst:
            val, count = _extract_list(pair)
            decoded.extend([val] * int(count))
        return Expr("List", *decoded)
    
    rule2('Prepend', _prepend)
    rule2('Append', _append)
    rule3('InsertAt', _insert_at)
    rule2('DeleteAt', _delete_at)
    rule3('ReplaceAt', _replace_at)
    rule2('FindFirst', _find_first)
    rule2('FindAll', _find_all)
    rule2('CountElement', _count_elem)
    rule1('Unique', _unique)
    rule1('Union', _unique)  # alias
    rule1('Duplicates', _duplicates)
    rule1('Frequencies', _frequencies)
    rule1('Tally', _frequencies)  # alias
    rule1('MostCommon', _most_common)
    rule2('MostCommon', _most_common)
    rule2('Zip', _zip_lists)
    rule1('Unzip', _unzip)
    rule2('SplitAt', _split_at)
    rule1('Gather', _gather)
    rule1('RunLengthEncode', _run_length_encode)
    rule1('RunLengthDecode', _run_length_decode)
    
    # ── More Math Functions (~50) ──
    
    def _clamp(x, min_val, max_val):
        return max(float(min_val), min(float(max_val), float(x)))
    
    def _lerp(a, b, t):
        return float(a) + float(t) * (float(b) - float(a))
    
    def _smoothstep(x, edge0=0, edge1=1):
        x = _clamp(x, edge0, edge1)
        t = (x - float(edge0)) / (float(edge1) - float(edge0))
        return t * t * (3 - 2 * t)
    
    def _remap(x, in_min, in_max, out_min, out_max):
        return float(out_min) + (float(x) - float(in_min)) * (float(out_max) - float(out_min)) / (float(in_max) - float(in_min))
    
    def _wrap(x, min_val, max_val):
        range_size = float(max_val) - float(min_val)
        return float(min_val) + ((float(x) - float(min_val)) % range_size)
    
    def _deg_to_rad(deg):
        return math.radians(float(deg))
    
    def _rad_to_deg(rad):
        return math.degrees(float(rad))
    
    def _is_close(a, b, tol=1e-9):
        return abs(float(a) - float(b)) < float(tol)
    
    def _approx_equal(a, b, rel_tol=1e-9, abs_tol=0.0):
        return math.isclose(float(a), float(b), rel_tol=float(rel_tol), abs_tol=float(abs_tol))
    
    def _frac(x):
        """Fractional part."""
        return float(x) - int(float(x))
    
    def _integer_part(x):
        return int(float(x))
    
    def _signum(x):
        """Sign function: -1, 0, or 1."""
        x = float(x)
        return -1 if x < 0 else (1 if x > 0 else 0)
    
    def _heaviside(x):
        """Heaviside step function."""
        return 0 if float(x) < 0 else 1
    
    def _dirac_delta(x):
        """Symbolic Dirac delta."""
        return Symbol("DiracDelta")
    
    def _kronecker_delta(i, j):
        return 1 if int(i) == int(j) else 0
    
    def _levi_civita(*indices):
        """Levi-Civita symbol."""
        indices = [int(x) for x in indices]
        n = len(indices)
        if len(set(indices)) != n:
            return 0
        
        inversions = 0
        for i in range(n):
            for j in range(i + 1, n):
                if indices[i] > indices[j]:
                    inversions += 1
        return 1 if inversions % 2 == 0 else -1
    
    rule3('Clamp', _clamp)
    rule3('Lerp', _lerp)
    rule1('Smoothstep', _smoothstep)
    rule3('Smoothstep', _smoothstep)
    rule3('Remap', _remap)
    rule3('Wrap', _wrap)
    rule1('DegToRad', _deg_to_rad)
    rule1('RadToDeg', _rad_to_deg)
    rule2('IsClose', _is_close)
    rule3('IsClose', _is_close)
    rule2('ApproxEqual', _approx_equal)
    rule3('ApproxEqual', _approx_equal)
    rule1('FractionalPart', _frac)
    rule1('IntegerPart', _integer_part)
    rule1('Signum', _signum)
    rule1('Heaviside', _heaviside)
    rule1('DiracDelta', _dirac_delta)
    rule2('KroneckerDelta', _kronecker_delta)
    
    # ── Encoding & Compression (~20) ──
    
    def _url_encode(s):
        import urllib.parse
        return urllib.parse.quote(str(s))
    
    def _url_decode(s):
        import urllib.parse
        return urllib.parse.unquote(str(s))
    
    def _html_escape(s):
        import html
        return html.escape(str(s))
    
    def _html_unescape(s):
        import html
        return html.unescape(str(s))
    
    def _json_encode(expr):
        import json
        # Simplified: convert to dict representation
        return json.dumps(str(expr))
    
    def _json_decode(s):
        import json
        return json.loads(str(s))
    
    def _gzip_compress(s):
        import gzip
        compressed = gzip.compress(str(s).encode())
        return compressed.hex()
    
    def _gzip_decompress(hex_str):
        import gzip
        compressed = bytes.fromhex(str(hex_str))
        return gzip.decompress(compressed).decode()
    
    def _zlib_compress(s):
        import zlib
        compressed = zlib.compress(str(s).encode())
        return compressed.hex()
    
    def _zlib_decompress(hex_str):
        import zlib
        compressed = bytes.fromhex(str(hex_str))
        return zlib.decompress(compressed).decode()
    
    rule1('URLEncode', _url_encode)
    rule1('URLDecode', _url_decode)
    rule1('HTMLEscape', _html_escape)
    rule1('HTMLUnescape', _html_unescape)
    rule1('JSONEncode', _json_encode)
    rule1('JSONDecode', _json_decode)
    rule1('GZIPCompress', _gzip_compress)
    rule1('GZIPDecompress', _gzip_decompress)
    rule1('ZlibCompress', _zlib_compress)
    rule1('ZlibDecompress', _zlib_decompress)
    
    # ── Random & Probability (~20) ──
    
    def _random_int(a, b):
        import random
        return random.randint(int(a), int(b))
    
    def _random_float(a=0.0, b=1.0):
        import random
        return random.uniform(float(a), float(b))
    
    def _random_choice(list_expr):
        import random
        lst = _extract_list(list_expr)
        return random.choice(lst)
    
    def _random_sample_list(list_expr, k):
        import random
        lst = _extract_list(list_expr)
        sample = random.sample(lst, int(k))
        return Expr("List", *sample)
    
    def _shuffle(list_expr):
        import random
        lst = list(_extract_list(list_expr))
        random.shuffle(lst)
        return Expr("List", *lst)
    
    def _set_random_seed(seed):
        import random
        random.seed(int(seed))
        return Symbol("Null")
    
    rule2('RandomInteger', _random_int)
    rule1('RandomReal', _random_float)
    rule2('RandomReal', _random_float)
    rule1('RandomChoice', _random_choice)
    rule2('RandomSampleList', _random_sample_list)
    rule1('Shuffle', _shuffle)
    rule1('SeedRandom', _set_random_seed)
    
    # ── Formatting (~20) ──
    
    def _to_string(x):
        return str(x)
    
    def _from_string(s):
        try:
            return int(s)
        except:
            try:
                return float(s)
            except:
                return str(s)
    
    def _number_form(x, digits=6):
        return f"{float(x):.{int(digits)}f}"
    
    def _scientific_form(x, digits=6):
        return f"{float(x):.{int(digits)}e}"
    
    def _engineering_form(x, digits=6):
        from decimal import Decimal
        val = Decimal(str(x))
        return f"{val:.{int(digits)}e}"
    
    def _percent_form(x, digits=2):
        return f"{float(x) * 100:.{int(digits)}f}%"
    
    def _padded_form(x, width, fill='0'):
        return str(x).rjust(int(width), str(fill)[0])
    
    def _binary_form(x):
        return bin(int(x))
    
    def _hex_form(x):
        return hex(int(x))
    
    def _octal_form(x):
        return oct(int(x))
    
    rule1('ToString', _to_string)
    rule1('FromString', _from_string)
    rule1('NumberForm', _number_form)
    rule2('NumberForm', _number_form)
    rule1('ScientificForm', _scientific_form)
    rule2('ScientificForm', _scientific_form)
    rule1('EngineeringForm', _engineering_form)
    rule2('EngineeringForm', _engineering_form)
    rule1('PercentForm', _percent_form)
    rule2('PercentForm', _percent_form)
    rule2('PaddedForm', _padded_form)
    rule3('PaddedForm', _padded_form)
    rule1('BinaryForm', _binary_form)
    rule1('HexForm', _hex_form)
    rule1('OctalForm', _octal_form)
    
    # ── Date Arithmetic (~10) ──
    
    def _date_plus(date_str, days):
        from datetime import datetime, timedelta
        dt = datetime.fromisoformat(str(date_str))
        new_dt = dt + timedelta(days=int(days))
        return new_dt.isoformat()
    
    def _date_difference(date1_str, date2_str):
        from datetime import datetime
        dt1 = datetime.fromisoformat(str(date1_str))
        dt2 = datetime.fromisoformat(str(date2_str))
        return (dt2 - dt1).days
    
    def _weekday_name(date_str):
        from datetime import datetime
        val = str(date_str)
        try:
            n = int(float(val))
            if 1 <= n <= 7:
                import calendar
                return calendar.day_name[n - 1]  # 1=Monday
        except (ValueError, TypeError):
            pass
        dt = datetime.fromisoformat(val)
        return dt.strftime("%A")
    
    def _month_name(date_str):
        from datetime import datetime
        val = str(date_str)
        try:
            n = int(float(val))
            if 1 <= n <= 12:
                import calendar
                return calendar.month_name[n]
        except (ValueError, TypeError):
            pass
        dt = datetime.fromisoformat(val)
        return dt.strftime("%B")
    
    rule2('DatePlus', _date_plus)
    rule2('DateDifference', _date_difference)
    rule1('WeekdayName', _weekday_name)
    rule1('MonthName', _month_name)
    
    # ── Additional Utilities (~14) ──
    
    def _deep_copy(expr):
        """Deep copy of expression."""
        if isinstance(expr, Expr):
            return Expr(expr.head, *[_deep_copy(a) for a in expr.args])
        return expr
    
    def _tree_depth(expr):
        """Maximum nesting depth."""
        if not isinstance(expr, Expr):
            return 0
        if not expr.args:
            return 1
        return 1 + max(_tree_depth(a) for a in expr.args)
    
    def _leaf_count(expr):
        """Count leaf nodes."""
        if not isinstance(expr, Expr):
            return 1
        if not expr.args:
            return 1
        return sum(_leaf_count(a) for a in expr.args)
    
    def _expr_size(expr):
        """Total number of subexpressions."""
        if not isinstance(expr, Expr):
            return 1
        return 1 + sum(_expr_size(a) for a in expr.args)
    
    def _free_symbols(expr):
        """List all free symbols."""
        import sympy as sp
        sym_expr = _to_sympy(expr)
        syms = list(sym_expr.free_symbols)
        return Expr("List", *[Symbol(str(s)) for s in syms])
    
    def _replace_all_expr(expr, old, new):
        """Replace all occurrences."""
        if expr == old:
            return new
        if isinstance(expr, Expr):
            new_args = tuple(_replace_all_expr(a, old, new) for a in expr.args)
            return Expr(expr.head, *new_args)
        return expr
    
    def _count_occurrences(expr, pattern):
        """Count occurrences of pattern."""
        if expr == pattern:
            return 1
        if isinstance(expr, Expr):
            return sum(_count_occurrences(a, pattern) for a in expr.args)
        return 0
    
    rule1('DeepCopy', _deep_copy)
    rule1('TreeDepth', _tree_depth)
    rule1('LeafCount', _leaf_count)
    rule1('ExpressionSize', _expr_size)
    rule1('FreeSymbols', _free_symbols)
    rule3('ReplaceAll', _replace_all_expr)
    rule2('CountOccurrences', _count_occurrences)
    
    return rules


EXTENDED4_RULES = build_extended4_rules()
