#!/usr/bin/env python3

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
from datetime import datetime

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

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm for progress bars...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

try:
    import networkx as nx
except ImportError:
    print("Installing networkx for GraphML export...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "networkx"])
    import networkx as nx


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
    
    def analyze_directory(self, directory: str, max_workers: int = None) -> None:
        """Analyze all Python files in a directory recursively using parallel processing."""
        # Collect all Python files first
        python_files = []
        for root, dirs, files in os.walk(directory):
            # Skip common directories that shouldn't be analyzed
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.venv', 'venv', 'node_modules'}]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        if not python_files:
            print("No Python files found in the specified directory.")
            return
        
        print(f"Found {len(python_files)} Python files to analyze")
        
        # Use parallel processing with progress bar
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), len(python_files))
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files for processing
            futures = {executor.submit(self._parse_file_worker, file_path): file_path 
                      for file_path in python_files}
            
            # Process results with progress bar
            results = []
            with tqdm(total=len(python_files), desc="Analyzing files", unit="file") as pbar:
                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            pbar.set_postfix_str(f"Last: {Path(file_path).name}")
                    except Exception as e:
                        print(f"\nError analyzing {file_path}: {e}")
                    finally:
                        pbar.update(1)
        
        # Merge results from all workers
        print("\nMerging results from parallel analysis...")
        for functions, calls in tqdm(results, desc="Merging"):
            # Merge functions
            self.functions.update(functions)
            # Merge calls
            for caller, callees in calls.items():
                self.calls[caller].update(callees)
    
    @staticmethod
    def _parse_file_worker(file_path: str):
        """Worker function for parallel file parsing."""
        try:
            # Create a new parser instance for this worker
            PY_LANGUAGE = Language(tspython.language())
            parser = Parser(PY_LANGUAGE)
            
            with open(file_path, 'rb') as f:
                source_code = f.read()
            
            tree = parser.parse(source_code)
            
            # Extract functions and calls
            functions = {}
            calls = defaultdict(set)
            
            # Helper to get module path
            def get_module_path(path):
                p = Path(path)
                if p.name == '__init__.py':
                    return str(p.parent).replace('/', '.')
                else:
                    return str(p.with_suffix('')).replace('/', '.')
            
            # Extract functions
            def extract_functions(node, functions_dict):
                if node.type == 'function_definition':
                    name_node = node.child_by_field_name('name')
                    if name_node:
                        func_name = source_code[name_node.start_byte:name_node.end_byte].decode('utf-8')
                        line_number = name_node.start_point[0] + 1
                        
                        module_path = get_module_path(file_path)
                        full_name = f"{module_path}.{func_name}" if module_path else func_name
                        functions_dict[full_name] = (file_path, line_number)
                        functions_dict[func_name] = (file_path, line_number)
                
                for child in node.children:
                    extract_functions(child, functions_dict)
            
            # Extract calls
            def extract_calls(node, calls_dict, current_function=None):
                if node.type == 'function_definition':
                    name_node = node.child_by_field_name('name')
                    if name_node:
                        current_function = source_code[name_node.start_byte:name_node.end_byte].decode('utf-8')
                
                if node.type == 'call':
                    function_node = node.child_by_field_name('function')
                    if function_node:
                        if function_node.type == 'identifier':
                            called_func = source_code[function_node.start_byte:function_node.end_byte].decode('utf-8')
                        elif function_node.type == 'attribute':
                            attr_node = function_node.child_by_field_name('attribute')
                            if attr_node:
                                called_func = source_code[attr_node.start_byte:attr_node.end_byte].decode('utf-8')
                            else:
                                called_func = source_code[function_node.start_byte:function_node.end_byte].decode('utf-8')
                        else:
                            called_func = source_code[function_node.start_byte:function_node.end_byte].decode('utf-8')
                        
                        if current_function:
                            calls_dict[current_function].add(called_func)
                        else:
                            module_name = get_module_path(file_path)
                            calls_dict[f"{module_name}.__main__"].add(called_func)
                
                for child in node.children:
                    extract_calls(child, calls_dict, current_function)
            
            extract_functions(tree.root_node, functions)
            extract_calls(tree.root_node, calls)
            
            return (functions, dict(calls))
            
        except Exception as e:
            raise Exception(f"Error processing {file_path}: {e}")
    
    def identify_external_calls(self) -> None:
        """Identify calls to functions not defined in the codebase."""
        all_called = set()
        for caller_funcs in self.calls.values():
            all_called.update(caller_funcs)
        
        defined_funcs = set(self.functions.keys())
        self.external_calls = all_called - defined_funcs
    
    def generate_networkx_graph(self, show_external: bool = True) -> nx.DiGraph:
        """Generate a NetworkX directed graph for the call flow."""
        G = nx.DiGraph()
        
        # Add nodes for all functions with metadata
        for func_name, (file_path, line_no) in self.functions.items():
            if '.' not in func_name:  # Skip duplicate entries with module paths
                G.add_node(func_name, 
                          file=Path(file_path).name,
                          full_path=file_path,
                          line=line_no,
                          type='internal_function')
        
        # Track module-level calls
        module_level_used = False
        
        # Add edges for function calls
        for caller, callees in self.calls.items():
            # Handle module-level calls
            if '.__main__' in caller:
                caller_display = 'module_level'
                if not G.has_node(caller_display):
                    G.add_node(caller_display, type='module_level')
                    module_level_used = True
            else:
                caller_display = caller
                if not G.has_node(caller_display):
                    # Add caller if not already in graph (might be from external module)
                    G.add_node(caller_display, type='function')
            
            for callee in callees:
                if callee in self.functions:
                    # Internal call
                    if G.has_node(callee):
                        G.add_edge(caller_display, callee, type='internal_call')
                elif show_external and callee not in {'print', 'len', 'range', 'str', 'int', 'float', 
                                                       'list', 'dict', 'set', 'tuple', 'bool', 'type'}:
                    # External call (not a builtin)
                    if not G.has_node(callee):
                        G.add_node(callee, type='external_function')
                    G.add_edge(caller_display, callee, type='external_call')
        
        return G
    
    def save_graphml(self, target_directory: str, graph_name: str = "call_graph", show_external: bool = True) -> str:
        """Save the call flow graph as a GraphML file in .call_graphs directory."""
        # Create .call_graphs directory in the target location
        if os.path.isfile(target_directory):
            target_directory = os.path.dirname(target_directory)
        
        graphs_dir = os.path.join(target_directory, '.call_graphs')
        os.makedirs(graphs_dir, exist_ok=True)
        
        # Generate the NetworkX graph
        G = self.generate_networkx_graph(show_external=show_external)
        
        # Add graph metadata
        G.graph['name'] = graph_name
        G.graph['total_functions'] = len(self.functions)
        G.graph['total_calls'] = sum(len(calls) for calls in self.calls.values())
        G.graph['analyzed_at'] = str(Path(target_directory).absolute())
        
        # Save as GraphML
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graphml_file = os.path.join(graphs_dir, f"{graph_name}_{timestamp}.graphml")
        
        nx.write_graphml(G, graphml_file)
        
        # Also save a latest version for easy access
        latest_file = os.path.join(graphs_dir, f"{graph_name}_latest.graphml")
        nx.write_graphml(G, latest_file)
        
        # Save analysis summary as JSON
        summary = {
            'timestamp': timestamp,
            'analyzed_directory': str(Path(target_directory).absolute()),
            'total_files': len(set(path for path, _ in self.functions.values())),
            'total_functions': len([f for f in self.functions.keys() if '.' not in f]),
            'total_calls': sum(len(calls) for calls in self.calls.values()),
            'total_external_calls': len(self.external_calls),
            'graph_files': {
                'graphml': os.path.basename(graphml_file),
                'graphml_latest': os.path.basename(latest_file)
            },
            'functions': {name: {'file': path, 'line': line} 
                         for name, (path, line) in self.functions.items() if '.' not in name},
            'call_relationships': {caller: list(callees) for caller, callees in self.calls.items()},
            'external_functions': list(self.external_calls)
        }
        
        summary_file = os.path.join(graphs_dir, f"{graph_name}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  Summary JSON: {summary_file}")
        
        return graphml_file
    
    def generate_dot_graph(self, show_external: bool = True) -> str:
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
                elif show_external and callee not in {'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set'}:
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate call flow graphs from Python code using tree-sitter')
    parser.add_argument('target', help='Directory or file to analyze')
    parser.add_argument('-o', '--output', default='call_flow_graph', help='Output file name (without extension)')
    parser.add_argument('-w', '--workers', type=int, default=None, 
                       help=f'Number of parallel workers (default: number of CPU cores, max {multiprocessing.cpu_count()})')
    parser.add_argument('--no-external', action='store_true', help='Hide external/undefined function calls in the graph')
    parser.add_argument('--format', choices=['svg', 'graphml', 'both'], default='both',
                       help='Output format: svg (Graphviz), graphml (NetworkX), or both (default: both)')
    parser.add_argument('--no-viz', action='store_true', help='Skip SVG visualization, only generate GraphML')
    
    args = parser.parse_args()
    
    if args.workers and args.workers > multiprocessing.cpu_count():
        print(f"Warning: Using {multiprocessing.cpu_count()} workers (system maximum)")
        args.workers = multiprocessing.cpu_count()
    
    print(f"🚀 Tree-Sitter Call Flow Analyzer")
    print(f"  Using {args.workers or multiprocessing.cpu_count()} parallel workers")
    print("-" * 50)
    
    analyzer = CallFlowAnalyzer(language='python')
    
    if os.path.isfile(args.target):
        print(f"Analyzing file: {args.target}")
        analyzer.parse_file(args.target)
    elif os.path.isdir(args.target):
        print(f"Analyzing directory: {args.target}")
        analyzer.analyze_directory(args.target, max_workers=args.workers)
    else:
        print(f"Error: {args.target} is not a valid file or directory")
        sys.exit(1)
    
    analyzer.identify_external_calls()
    analyzer.print_summary()
    
    # Save GraphML format (always saves to .call_graphs directory)
    if args.format in ['graphml', 'both']:
        try:
            graphml_path = analyzer.save_graphml(
                args.target, 
                graph_name=args.output,
                show_external=not args.no_external
            )
            print(f"\n✓ GraphML saved to: {graphml_path}")
            print(f"  Latest version: {os.path.join(os.path.dirname(graphml_path), f'{args.output}_latest.graphml')}")
        except Exception as e:
            print(f"\nWarning: Could not save GraphML: {e}")
    
    # Generate SVG visualization
    if args.format in ['svg', 'both'] and not args.no_viz:
        try:
            dot_graph = analyzer.generate_dot_graph(show_external=not args.no_external)
            dot_graph.render(args.output, cleanup=True)
            print(f"\n✓ SVG visualization saved to: {args.output}.svg")
            print(f"  Open this file in a browser to view the interactive graph")
        except Exception as e:
            print(f"\nWarning: Could not generate graph visualization: {e}")
            print("You may need to install Graphviz on your system:")
            print("  - macOS: brew install graphviz")
            print("  - Ubuntu/Debian: sudo apt-get install graphviz")
            print("  - Windows: Download from https://graphviz.org/download/")


if __name__ == "__main__":
    main()