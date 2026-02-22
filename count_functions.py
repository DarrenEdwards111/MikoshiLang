#!/usr/bin/env python3
"""
Simple function counter - count all function patterns in all modules
"""
import os
import re

def count_functions_in_file(filepath):
    """Count function patterns in a Python file"""
    functions = set()
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find all patterns like ('FunctionName[...]', lambda ...)
    # Look for quoted strings followed by lambda
    pattern = r"\('([A-Za-z][A-Za-z0-9]*)\[.*?\]',\s*lambda"
    matches = re.findall(pattern, content)
    
    for match in matches:
        functions.add(match)
    
    return functions

def main():
    mikoshilang_dir = '/home/darre/.openclaw/workspace/mikoshilang/mikoshilang'
    
    all_functions = set()
    file_counts = {}
    
    # Check all .py files in mikoshilang directory
    for filename in sorted(os.listdir(mikoshilang_dir)):
        if filename.endswith('.py') and not filename.startswith('__'):
            filepath = os.path.join(mikoshilang_dir, filename)
            functions = count_functions_in_file(filepath)
            
            if functions:
                file_counts[filename] = len(functions)
                all_functions.update(functions)
                print(f"{filename:30s} {len(functions):4d} functions")
    
    print("\n" + "=" * 60)
    print(f"TOTAL UNIQUE FUNCTIONS: {len(all_functions)}")
    print("=" * 60)
    
    # Show breakdown by category
    print("\nBreakdown:")
    for filename, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {filename:30s} {count:4d}")
    
    return len(all_functions)

if __name__ == '__main__':
    total = main()
