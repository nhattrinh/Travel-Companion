#!/usr/bin/env python3
"""
Envelope consistency validation script.
Scans all endpoint handlers to verify envelope pattern compliance.
Implements T125/UX001 - Envelope consistency check.
"""
import re
import sys
from pathlib import Path
from typing import List, Tuple


def extract_return_statements(file_path: Path) -> List[Tuple[int, str, str]]:
    """Extract return statements from endpoint handlers only."""
    returns = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    in_endpoint = False
    in_class = False
    function_name = ""
    indent_level = 0
    in_return = False
    return_lines = []
    return_start_line = 0
    
    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()
        
        # Track class definitions
        if stripped.startswith('class '):
            in_class = True
            in_endpoint = False
            continue
        
        # Track endpoint function definitions (decorated with @router)
        if '@router.' in stripped:
            # Next line should be the function definition
            in_endpoint = True
            continue
        
        # Capture function definition after @router decorator
        if in_endpoint and (stripped.startswith('async def ') or stripped.startswith('def ')):
            function_name = stripped.split('(')[0].replace('async def ', '').replace('def ', '')
            indent_level = len(line) - len(line.lstrip())
            continue
        
        # Handle multi-line returns
        if in_return:
            return_lines.append(stripped)
            # Check if we've completed the return statement (closing brace or semicolon)
            if '}' in stripped or stripped.endswith(',') == False:
                # Join all lines of the return statement
                full_return = ' '.join(return_lines)
                returns.append((return_start_line, full_return, function_name))
                in_return = False
                return_lines = []
            continue
        
        # Check for return statements in endpoint functions only
        if in_endpoint and stripped.startswith('return'):
            current_indent = len(line) - len(line.lstrip())
            
            # Only capture returns at the same or deeper indent than function def
            if current_indent > indent_level:
                return_start_line = line_num
                return_lines = [stripped]
                
                # Check if this is a single-line return
                if '{' in stripped and '}' in stripped:
                    returns.append((line_num, stripped, function_name))
                    return_lines = []
                elif '{' in stripped:
                    # Multi-line dict return
                    in_return = True
                else:
                    # Simple value return
                    returns.append((line_num, stripped, function_name))
                    return_lines = []
        
        # Reset when we exit the function (dedent back to module/class level)
        if in_endpoint and stripped and not stripped.startswith('#'):
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level:
                in_endpoint = False
                function_name = ""
                in_return = False
                return_lines = []
    
    return returns


def check_envelope_format(return_stmt: str) -> bool:
    """
    Check if a return statement matches the envelope pattern.
    
    Envelope pattern: {"status": "ok"|"error", "data": ..., "error": ...}
    """
    # Skip non-dict returns
    if 'return {' not in return_stmt and 'return JSONResponse' not in return_stmt:
        return True  # Not a dict return, skip validation
    
    # Check for envelope keys
    has_status = '"status"' in return_stmt or "'status'" in return_stmt
    has_data = '"data"' in return_stmt or "'data'" in return_stmt
    has_error = '"error"' in return_stmt or "'error'" in return_stmt
    
    # Valid envelope must have all three keys
    if has_status and has_data and has_error:
        return True
    
    # If it has any envelope key but not all, it's inconsistent
    if has_status or has_data or has_error:
        return False
    
    # If it has none, it might be a helper/internal function
    # Check if it's likely an endpoint (has Response/dict return)
    if '{' in return_stmt:
        return False  # Dict without envelope keys = inconsistent
    
    return True  # Non-dict return, skip


def scan_endpoint_file(file_path: Path) -> List[Tuple[int, str, str]]:
    """Scan an endpoint file for envelope consistency issues."""
    issues = []
    returns = extract_return_statements(file_path)
    
    for line_num, return_stmt, function_name in returns:
        if not check_envelope_format(return_stmt):
            issues.append((line_num, return_stmt, function_name))
    
    return issues


def main():
    """Main validation script."""
    project_root = Path(__file__).parent.parent
    api_dir = project_root / "app" / "api"
    
    if not api_dir.exists():
        print(f"Error: API directory not found at {api_dir}")
        sys.exit(1)
    
    print("🔍 Scanning endpoints for envelope consistency...\n")
    
    # Find all endpoint files
    endpoint_files = list(api_dir.glob("*_endpoints.py"))
    
    if not endpoint_files:
        print("Warning: No endpoint files found")
        sys.exit(0)
    
    total_issues = 0
    files_with_issues = 0
    
    for endpoint_file in sorted(endpoint_files):
        issues = scan_endpoint_file(endpoint_file)
        
        if issues:
            files_with_issues += 1
            total_issues += len(issues)
            
            print(f"❌ {endpoint_file.name}:")
            for line_num, return_stmt, function_name in issues:
                print(f"   {function_name}() line {line_num}: {return_stmt[:80]}...")
            print()
    
    # Summary
    print("=" * 60)
    print(f"📊 Scanned {len(endpoint_files)} endpoint files")
    print(f"   Files with issues: {files_with_issues}")
    print(f"   Total violations: {total_issues}")
    print("=" * 60)
    
    if total_issues > 0:
        print("\n⚠️  Envelope consistency check FAILED")
        print("\nExpected envelope format:")
        print('  {"status": "ok"|"error", "data": <payload>|null, "error": null|<message>}')
        sys.exit(1)
    else:
        print("\n✅ All endpoints use consistent envelope format")
        sys.exit(0)


if __name__ == "__main__":
    main()
