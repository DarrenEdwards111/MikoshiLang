#!/usr/bin/env python3
"""
Test a large sample of functions from each category
"""
from mikoshilang import parse, evaluate

# Sample functions from each extended module
test_functions = {
    'Core Math': [
        ('Plus', '2, 3'),
        ('Sin', 'Pi'),
        ('Factorial', '5'),
        ('Sqrt', '16'),
    ],
    'Extended6-9': [
        ('Gradient', 'x^2, {x}'),
        ('LinearRegression', '{1,2,3}, {2,4,6}'),  # Simple 1D regression
        ('KineticEnergy', '2, 3'),
        ('DNAComplement', '"ATCG"'),
    ],
    'Extended10-12': [
        ('BesselJ', '0, 1'),
        ('FFT', '{1,2,3,4}'),
        ('ShortestPath', '{{1,2},{2,3}}, 1, 3'),
    ],
    'Extended13-16': [
        ('BubbleSort', '{3,1,2}'),
        ('SemiMajorAxis', '1e11, 3e7'),
        ('TTest', '{1,2,3}, {2,3,4}'),
        ('Tokenize', '"hello world"'),
    ],
    'Extended17-20': [
        ('OhmsLaw', '12, 2'),
        ('GeneExpression', '100'),
        ('BlackScholes', '100, 100, 0.05, 1, 0.2'),
        ('ReynoldsNumber', '1000, 1, 0.001, 0.01'),
    ],
    'Extended21-24': [
        ('GreenhouseEffect', '255, 0.7'),
        ('BMI', '70, 1.75'),
        ('DHTransform', '0, 0, 1, 0'),
        ('CarbonSequestration', '100, 10'),
    ],
    'Extended25-28': [
        ('BravaisLattice', '"cubic"'),
        ('SnellsLawOptics', '1, 1.5, 0.5'),
        ('LiftForceAero', '1.225, 100, 20, 0.5'),
        ('NuclearBindingEnergy', '235, 92'),
    ],
    'Extended29-33': [
        ('EntropyShannon', '{0.5, 0.5}'),
        ('SoundPressureLevel', '1, 0.00002'),
        ('HaversineDistance', '0, 0, 0.1, 0.1, 6371'),
        ('IdealGasEquation', '101325, 0.0224, 1, 8.314, 273'),
        ('SoilMoisture', '0.3'),
    ],
}

def test_function(name, args):
    """Test if function works"""
    try:
        expr = parse(f"{name}[{args}]")
        result = evaluate(expr)
        result_str = str(result)
        
        # Check if it returned the input (placeholder)
        if result_str == f"{name}[{args}]":
            return 'placeholder'
        return 'works'
    except Exception as e:
        return 'error'

print("=" * 80)
print("SAMPLING FUNCTIONS FROM EACH MODULE")
print("=" * 80)

total = 0
working = 0
placeholder = 0
error = 0

for category, functions in test_functions.items():
    print(f"\n{category}:")
    cat_work = 0
    cat_place = 0
    cat_err = 0
    
    for name, args in functions:
        total += 1
        status = test_function(name, args)
        
        if status == 'works':
            working += 1
            cat_work += 1
            symbol = '✓'
        elif status == 'placeholder':
            placeholder += 1
            cat_place += 1
            symbol = '○'
        else:
            error += 1
            cat_err += 1
            symbol = '✗'
        
        print(f"  {symbol} {name}[{args[:30]}...]")
    
    print(f"  ({cat_work} working, {cat_place} placeholder, {cat_err} errors)")

print("\n" + "=" * 80)
print("SAMPLE RESULTS")
print("=" * 80)
print(f"Total tested:    {total}")
print(f"✓ Working:       {working} ({working/total*100:.1f}%)")
print(f"○ Placeholders:  {placeholder} ({placeholder/total*100:.1f}%)")
print(f"✗ Errors:        {error} ({error/total*100:.1f}%)")

print("\n" + "=" * 80)
print("ACTUAL FUNCTION COUNT")
print("=" * 80)
print("\nFrom source code analysis:")
print(f"  Base extended (1-5):  1,269 functions")
print(f"  Extended (6-33):      4,455 functions")
print(f"  TOTAL:                5,724 functions")
print(f"\nREADME claims: 6,094 functions")
print(f"Actual count:  5,724 functions")
print(f"Difference:    -370 functions")
print("\nThe README is overcounting by ~370 functions.")
print("Need to update README from 6,094 to 5,724.")
