#!/usr/bin/env python3
"""
Comprehensive test of ALL MikoshiLang functions
Automatically extracts and tests every registered function
"""
import sys
import re
from mikoshilang import parse, evaluate

def extract_all_functions():
    """Extract all function patterns from all modules"""
    functions = {}
    
    # Import all extended modules and get their rules
    modules_to_check = []
    
    # Core modules
    try:
        from mikoshilang.extended import EXTENDED_RULES
        modules_to_check.append(('extended', EXTENDED_RULES))
    except:
        pass
    
    try:
        from mikoshilang.extended2 import EXTENDED2_RULES
        modules_to_check.append(('extended2', EXTENDED2_RULES))
    except:
        pass
    
    try:
        from mikoshilang.extended3 import EXTENDED3_RULES
        modules_to_check.append(('extended3', EXTENDED3_RULES))
    except:
        pass
    
    try:
        from mikoshilang.extended4 import EXTENDED4_RULES
        modules_to_check.append(('extended4', EXTENDED4_RULES))
    except:
        pass
    
    try:
        from mikoshilang.extended5 import EXTENDED5_RULES
        modules_to_check.append(('extended5', EXTENDED5_RULES))
    except:
        pass
    
    # Extended6-33
    for i in range(6, 34):
        try:
            mod = __import__(f'mikoshilang.extended{i}', fromlist=['register'])
            rules = mod.register()
            modules_to_check.append((f'extended{i}', rules))
        except Exception as e:
            pass
    
    # Combine all rules
    all_rules = []
    for name, rules in modules_to_check:
        all_rules.extend(rules)
    
    for pattern, _ in all_rules:
        # Extract function name from pattern
        # Patterns look like: 'FunctionName[args_]' or 'FunctionName[x_, y_]'
        match = re.match(r'([A-Za-z][A-Za-z0-9]*)\[', pattern)
        if match:
            func_name = match.group(1)
            # Count number of arguments
            args = pattern[len(func_name)+1:-1]  # Everything between [ and ]
            if not args:
                arg_count = 0
            else:
                # Count underscores as argument placeholders
                arg_count = args.count('_')
                if arg_count == 0 and args:
                    arg_count = 1  # Has args but no blanks
            
            if func_name not in functions:
                functions[func_name] = arg_count
    
    return functions

def generate_test_input(arg_count):
    """Generate appropriate test inputs for given argument count"""
    if arg_count == 0:
        return ""
    elif arg_count == 1:
        return "2"
    elif arg_count == 2:
        return "2, 3"
    elif arg_count == 3:
        return "2, 3, 4"
    elif arg_count == 4:
        return "2, 3, 4, 5"
    else:
        return ", ".join(str(i) for i in range(1, arg_count + 1))

def test_function(func_name, args):
    """Test if a function is implemented or just a placeholder"""
    try:
        if args:
            expr_str = f"{func_name}[{args}]"
        else:
            expr_str = f"{func_name}[]"
        
        expr = parse(expr_str)
        result = evaluate(expr)
        result_str = str(result)
        
        # Check if it's just returning the input (placeholder)
        if result_str == expr_str:
            return ('placeholder', None)
        
        # Check for error messages
        if 'Error[' in result_str or 'error' in result_str.lower():
            return ('error', result_str)
        
        return ('implemented', result_str)
    
    except Exception as e:
        return ('exception', str(e))

def run_comprehensive_test():
    """Test all functions and report statistics"""
    print("=" * 80)
    print("EXTRACTING ALL MIKOSHILANG FUNCTIONS...")
    print("=" * 80)
    
    functions = extract_all_functions()
    total = len(functions)
    
    print(f"\nFound {total} unique functions")
    print("\nTesting all functions...\n")
    
    results = {
        'implemented': [],
        'placeholder': [],
        'error': [],
        'exception': []
    }
    
    # Test each function
    for i, (func_name, arg_count) in enumerate(sorted(functions.items()), 1):
        args = generate_test_input(arg_count)
        status, output = test_function(func_name, args)
        results[status].append({
            'name': func_name,
            'args': args,
            'output': output
        })
        
        # Progress indicator every 100 functions
        if i % 100 == 0:
            print(f"Tested {i}/{total} functions...")
    
    # Print results
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST RESULTS")
    print("=" * 80)
    print(f"\nTotal functions: {total}")
    print(f"\n✓ Implemented:  {len(results['implemented'])} ({len(results['implemented'])/total*100:.1f}%)")
    print(f"○ Placeholders: {len(results['placeholder'])} ({len(results['placeholder'])/total*100:.1f}%)")
    print(f"✗ Errors:       {len(results['error'])} ({len(results['error'])/total*100:.1f}%)")
    print(f"⚠ Exceptions:   {len(results['exception'])} ({len(results['exception'])/total*100:.1f}%)")
    
    # Show sample implemented functions
    if results['implemented']:
        print(f"\n✓ Sample working functions (showing first 20):")
        for func in results['implemented'][:20]:
            args_display = f"[{func['args']}]" if func['args'] else "[]"
            output = func['output'][:60] + "..." if len(func['output']) > 60 else func['output']
            print(f"  {func['name']}{args_display} → {output}")
    
    # Show sample placeholders
    if results['placeholder']:
        print(f"\n○ Sample placeholder functions (showing first 20):")
        for func in results['placeholder'][:20]:
            args_display = f"[{func['args']}]" if func['args'] else "[]"
            print(f"  {func['name']}{args_display}")
    
    # Show errors if any
    if results['error']:
        print(f"\n✗ Functions with errors (showing first 10):")
        for func in results['error'][:10]:
            args_display = f"[{func['args']}]" if func['args'] else "[]"
            output = func['output'][:100] if func['output'] else "Unknown error"
            print(f"  {func['name']}{args_display} → {output}")
    
    # Show exceptions if any
    if results['exception']:
        print(f"\n⚠ Functions with exceptions (showing first 10):")
        for func in results['exception'][:10]:
            args_display = f"[{func['args']}]" if func['args'] else "[]"
            output = func['output'][:100] if func['output'] else "Unknown exception"
            print(f"  {func['name']}{args_display} → {output}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    working = len(results['implemented'])
    not_working = total - working
    
    print(f"\n✓ Working functions:      {working} ({working/total*100:.1f}%)")
    print(f"○ Not implemented:        {not_working} ({not_working/total*100:.1f}%)")
    print(f"\nImplementation coverage:  {working/total*100:.1f}%")
    
    if working/total >= 0.50:
        print("\n✅ PASS: More than 50% of functions are implemented")
        return 0
    else:
        print("\n⚠️  WARNING: Less than 50% of functions are implemented")
        return 1

if __name__ == '__main__':
    exit_code = run_comprehensive_test()
    sys.exit(exit_code)
