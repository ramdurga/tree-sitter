#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from collections import defaultdict
import subprocess

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser, Node
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tree-sitter", "tree-sitter-python"])
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser, Node

try:
    import graphviz
except ImportError:
    print("Installing graphviz for visualization...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "graphviz"])
    import graphviz


class CallFlowAnalyzer:
    def __init__(self, language='python'):
        self.language = language
        
        if language == 'python':
            PY_LANGUAGE = Language(tspython.language())
            self.parser = Parser(PY_LANGUAGE)
        else:
            raise NotImplementedError(f"Language {language} not yet supported")
        
        self.functions = {}  # function_name -> (file_path, line_number)
        self.calls = defaultdict(set)  # caller -> set of callees
        self.external_calls = set()  # calls to functions not defined in codebase
        
    def parse_file(self, file_path: str) -> None:
        """Parse a single file and extract function definitions and calls."""
        with open(file_path, 'rb') as f:
            source_code = f.read()
        
        tree = self.parser.parse(source_code)
        self._extract_functions(tree.root_node, file_path, source_code)
        self._extract_calls(tree.root_node, file_path, source_code)
    
    def _extract_functions(self, node: Node, file_path: str, source_code: bytes) -> None:
        """Extract function definitions from the AST."""
        if node.type == 'function_definition':
            name_node = node.child_by_field_name('name')
            if name_node:
                func_name = source_code[name_node.start_byte:name_node.end_byte].decode('utf-8')
                line_number = name_node.start_point[0] + 1
                
                # Store with module path for better identification
                module_path = self._get_module_path(file_path)
                full_name = f"{module_path}.{func_name}" if module_path else func_name
                self.functions[full_name] = (file_path, line_number)
                
                # Also store without module path for local references
                self.functions[func_name] = (file_path, line_number)
        
        for child in node.children:
            self._extract_functions(child, file_path, source_code)
    
    def _extract_calls(self, node: Node, file_path: str, source_code: bytes, 
                      current_function: str = None) -> None:
        """Extract function calls from the AST."""
        # Track which function we're currently inside
        if node.type == 'function_definition':
            name_node = node.child_by_field_name('name')
            if name_node:
                current_function = source_code[name_node.start_byte:name_node.end_byte].decode('utf-8')
        
        # Detect function calls
        if node.type == 'call':
            function_node = node.child_by_field_name('function')
            if function_node:
                # Handle different call types
                if function_node.type == 'identifier':
                    # Simple function call: func()
                    called_func = source_code[function_node.start_byte:function_node.end_byte].decode('utf-8')
                elif function_node.type == 'attribute':
                    # Method call: obj.method()
                    attr_node = function_node.child_by_field_name('attribute')
                    if attr_node:
                        called_func = source_code[attr_node.start_byte:attr_node.end_byte].decode('utf-8')
                    else:
                        called_func = source_code[function_node.start_byte:function_node.end_byte].decode('utf-8')
                else:
                    # Complex call expression
                    called_func = source_code[function_node.start_byte:function_node.end_byte].decode('utf-8')
                
                if current_function:
                    self.calls[current_function].add(called_func)
                else:
                    # Call at module level
                    module_name = self._get_module_path(file_path)
                    self.calls[f"{module_name}.__main__"].add(called_func)
        
        for child in node.children:
            self._extract_calls(child, file_path, source_code, current_function)
    
    def _get_module_path(self, file_path: str) -> str:
        """Convert file path to module path."""
        path = Path(file_path)
        if path.name == '__init__.py':
            return str(path.parent).replace('/', '.')
        else:
            return str(path.with_suffix('')).replace('/', '.')
    
    def analyze_directory(self, directory: str) -> None:
        """Analyze all Python files in a directory recursively."""
        for root, dirs, files in os.walk(directory):
            # Skip common directories that shouldn't be analyzed
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.venv', 'venv', 'node_modules'}]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        print(f"Analyzing: {file_path}")
                        self.parse_file(file_path)
                    except Exception as e:
                        print(f"Error analyzing {file_path}: {e}")
    
    def identify_external_calls(self) -> None:
        """Identify calls to functions not defined in the codebase."""
        all_called = set()
        for caller_funcs in self.calls.values():
            all_called.update(caller_funcs)
        
        defined_funcs = set(self.functions.keys())
        self.external_calls = all_called - defined_funcs
    
    def generate_dot_graph(self) -> str:
        """Generate a DOT graph representation of the call flow."""
        dot = graphviz.Digraph(comment='Call Flow Graph', format='svg')
        dot.attr(rankdir='TB')
        
        # Add nodes for all functions
        for func_name, (file_path, line_no) in self.functions.items():
            if '.' not in func_name:  # Skip duplicate entries with module paths
                label = f"{func_name}\\n{Path(file_path).name}:{line_no}"
                dot.node(func_name, label, shape='box', style='filled', fillcolor='lightblue')
        
        # Add edges for function calls
        for caller, callees in self.calls.items():
            # Clean up caller name
            if '.__main__' in caller:
                caller_display = 'module_level'
                dot.node(caller_display, 'Module Level', shape='diamond', style='filled', fillcolor='yellow')
            else:
                caller_display = caller
            
            for callee in callees:
                if callee in self.functions:
                    # Internal call
                    dot.edge(caller_display, callee, color='black')
                elif callee not in {'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set'}:
                    # External call (not a builtin)
                    dot.node(callee, callee, shape='ellipse', style='dashed', color='gray')
                    dot.edge(caller_display, callee, color='gray', style='dashed')
        
        return dot
    
    def print_summary(self) -> None:
        """Print a summary of the analysis."""
        print("\n" + "="*60)
        print("CALL FLOW ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nTotal functions found: {len(self.functions)}")
        print(f"Total function calls tracked: {sum(len(calls) for calls in self.calls.values())}")
        
        print("\n--- Functions Defined ---")
        for func_name, (file_path, line_no) in sorted(self.functions.items()):
            if '.' not in func_name:  # Skip duplicates
                print(f"  {func_name:30} -> {file_path}:{line_no}")
        
        print("\n--- Call Relationships ---")
        for caller, callees in sorted(self.calls.items()):
            if callees:
                print(f"\n  {caller} calls:")
                for callee in sorted(callees):
                    if callee in self.functions:
                        print(f"    -> {callee}")
                    else:
                        print(f"    -> {callee} (external)")
        
        if self.external_calls:
            print("\n--- External/Undefined Functions Called ---")
            for ext_call in sorted(self.external_calls):
                if ext_call not in {'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set'}:
                    print(f"  - {ext_call}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python call_flow_graph.py <directory_or_file> [output_file]")
        print("\nExample:")
        print("  python call_flow_graph.py ./my_project")
        print("  python call_flow_graph.py ./my_project call_graph.svg")
        sys.exit(1)
    
    target_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "call_flow_graph"
    
    analyzer = CallFlowAnalyzer(language='python')
    
    if os.path.isfile(target_path):
        print(f"Analyzing file: {target_path}")
        analyzer.parse_file(target_path)
    elif os.path.isdir(target_path):
        print(f"Analyzing directory: {target_path}")
        analyzer.analyze_directory(target_path)
    else:
        print(f"Error: {target_path} is not a valid file or directory")
        sys.exit(1)
    
    analyzer.identify_external_calls()
    analyzer.print_summary()
    
    # Generate and save the graph
    try:
        dot_graph = analyzer.generate_dot_graph()
        dot_graph.render(output_file, cleanup=True)
        print(f"\n✓ Call flow graph saved to: {output_file}.svg")
        print(f"  Open this file in a browser to view the interactive graph")
    except Exception as e:
        print(f"\nWarning: Could not generate graph visualization: {e}")
        print("You may need to install Graphviz on your system:")
        print("  - macOS: brew install graphviz")
        print("  - Ubuntu/Debian: sudo apt-get install graphviz")
        print("  - Windows: Download from https://graphviz.org/download/")


if __name__ == "__main__":
    main()