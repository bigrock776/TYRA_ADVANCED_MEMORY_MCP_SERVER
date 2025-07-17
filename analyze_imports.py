#!/usr/bin/env python3
"""
Comprehensive import dependency analysis for Tyra MCP Memory Server.

This script analyzes all Python files and their import dependencies to:
1. Map all import relationships 
2. Identify orphaned modules
3. Detect circular dependencies
4. Generate dependency graph
5. Find unused imports
"""

import ast
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import json
import re

@dataclass
class ImportInfo:
    """Information about an import statement."""
    module: str  # Full module name
    names: List[str]  # Imported names (empty for 'import module')
    alias: Optional[str] = None  # Alias if any
    line_number: int = 0
    is_from_import: bool = False
    level: int = 0  # Relative import level (0 for absolute)

@dataclass
class FileAnalysis:
    """Analysis results for a single file."""
    file_path: str
    imports: List[ImportInfo] = field(default_factory=list)
    internal_imports: List[ImportInfo] = field(default_factory=list)  # Only internal project imports
    external_imports: List[ImportInfo] = field(default_factory=list)  # External library imports
    syntax_errors: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    is_entry_point: bool = False
    module_name: str = ""

class ImportAnalyzer:
    """Comprehensive import dependency analyzer."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.files: Dict[str, FileAnalysis] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.orphaned_modules: Set[str] = set()
        self.circular_dependencies: List[List[str]] = []
        self.entry_points = [
            "main.py",
            "src/mcp/server.py", 
            "src/api/app.py"
        ]
        
    def analyze_all_files(self) -> Dict[str, FileAnalysis]:
        """Analyze all Python files in the project."""
        python_files = list(self.project_root.glob("**/*.py"))
        
        print(f"Found {len(python_files)} Python files to analyze...")
        
        for file_path in python_files:
            try:
                relative_path = file_path.relative_to(self.project_root)
                analysis = self._analyze_file(file_path)
                analysis.file_path = str(relative_path)
                analysis.module_name = self._path_to_module_name(str(relative_path))
                analysis.is_entry_point = str(relative_path) in self.entry_points
                self.files[str(relative_path)] = analysis
                
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                
        self._build_dependency_graph()
        self._find_orphaned_modules()
        self._find_circular_dependencies()
        
        return self.files
    
    def _analyze_file(self, file_path: Path) -> FileAnalysis:
        """Analyze a single Python file."""
        analysis = FileAnalysis(file_path=str(file_path))
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_info = ImportInfo(
                            module=alias.name,
                            names=[],
                            alias=alias.asname,
                            line_number=node.lineno,
                            is_from_import=False
                        )
                        analysis.imports.append(import_info)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        names = [alias.name for alias in node.names]
                        import_info = ImportInfo(
                            module=node.module,
                            names=names,
                            line_number=node.lineno,
                            is_from_import=True,
                            level=node.level
                        )
                        analysis.imports.append(import_info)
                        
                elif isinstance(node, ast.FunctionDef):
                    analysis.functions.append(node.name)
                    
                elif isinstance(node, ast.ClassDef):
                    analysis.classes.append(node.name)
                    
        except SyntaxError as e:
            analysis.syntax_errors.append(str(e))
        except Exception as e:
            analysis.syntax_errors.append(f"Parse error: {e}")
            
        # Classify imports as internal or external
        for import_info in analysis.imports:
            if self._is_internal_import(import_info):
                analysis.internal_imports.append(import_info)
            else:
                analysis.external_imports.append(import_info)
                
        return analysis
    
    def _is_internal_import(self, import_info: ImportInfo) -> bool:
        """Check if an import is internal to the project."""
        module = import_info.module
        
        # Relative imports are internal
        if import_info.level > 0:
            return True
            
        # Check if module starts with project-specific prefixes
        internal_prefixes = [
            "src.",
            "core.",
            "api.",
            "mcp.",
            "clients.",
            "dashboard.",
            "memory.",
            "agents.",
            "ingest.",
            "migrations.",
            "suggestions.",
            "validators.",
            "infrastructure."
        ]
        
        return any(module.startswith(prefix) for prefix in internal_prefixes)
    
    def _path_to_module_name(self, file_path: str) -> str:
        """Convert file path to module name."""
        # Remove .py extension and convert slashes to dots
        module_name = file_path.replace('.py', '').replace('/', '.')
        
        # Remove __init__ from module names
        if module_name.endswith('.__init__'):
            module_name = module_name[:-9]
            
        return module_name
    
    def _build_dependency_graph(self):
        """Build dependency graph between modules."""
        for file_path, analysis in self.files.items():
            source_module = analysis.module_name
            
            for import_info in analysis.internal_imports:
                target_module = self._resolve_import_module(import_info, source_module)
                if target_module:
                    self.dependency_graph[source_module].add(target_module)
                    self.reverse_dependency_graph[target_module].add(source_module)
    
    def _resolve_import_module(self, import_info: ImportInfo, source_module: str) -> Optional[str]:
        """Resolve relative imports to absolute module names."""
        if import_info.level == 0:
            # Absolute import
            return import_info.module
        else:
            # Relative import
            source_parts = source_module.split('.')
            
            # Go up the hierarchy based on level
            if import_info.level >= len(source_parts):
                return None
                
            base_parts = source_parts[:-import_info.level]
            
            if import_info.module:
                return '.'.join(base_parts + [import_info.module])
            else:
                return '.'.join(base_parts)
    
    def _find_orphaned_modules(self):
        """Find modules that are never imported."""
        all_modules = set(analysis.module_name for analysis in self.files.values())
        imported_modules = set()
        
        for analysis in self.files.values():
            for import_info in analysis.internal_imports:
                resolved = self._resolve_import_module(import_info, analysis.module_name)
                if resolved:
                    imported_modules.add(resolved)
                    
        self.orphaned_modules = all_modules - imported_modules
    
    def _find_circular_dependencies(self):
        """Find circular dependencies using DFS."""
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> bool:
            if node in rec_stack:
                # Found a cycle - find where it starts
                try:
                    cycle_start = path.index(node)
                    cycle = path[cycle_start:] + [node]
                except ValueError:
                    # node not in path, add it
                    cycle = path + [node]
                if cycle not in self.circular_dependencies:
                    self.circular_dependencies.append(cycle)
                return True
                
            if node in visited:
                return False
                
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.dependency_graph.get(node, set()):
                if dfs(neighbor, path + [node]):
                    return True
                    
            rec_stack.remove(node)
            return False
        
        for module in self.dependency_graph:
            if module not in visited:
                dfs(module, [])
    
    def get_import_tree(self, root_module: str, max_depth: int = 3) -> Dict[str, Any]:
        """Get import tree starting from a root module."""
        visited = set()
        
        def build_tree(module: str, depth: int) -> Dict[str, Any]:
            if depth > max_depth or module in visited:
                return {"module": module, "depth": depth, "children": []}
                
            visited.add(module)
            
            children = []
            for imported_module in self.dependency_graph.get(module, set()):
                children.append(build_tree(imported_module, depth + 1))
                
            return {
                "module": module,
                "depth": depth,
                "children": children,
                "import_count": len(self.dependency_graph.get(module, set()))
            }
        
        return build_tree(root_module, 0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        total_files = len(self.files)
        total_imports = sum(len(analysis.imports) for analysis in self.files.values())
        internal_imports = sum(len(analysis.internal_imports) for analysis in self.files.values())
        external_imports = sum(len(analysis.external_imports) for analysis in self.files.values())
        
        # Count unique external libraries
        external_libraries = set()
        for analysis in self.files.values():
            for import_info in analysis.external_imports:
                external_libraries.add(import_info.module.split('.')[0])
        
        # Find most imported modules
        import_counts = defaultdict(int)
        for analysis in self.files.values():
            for import_info in analysis.internal_imports:
                resolved = self._resolve_import_module(import_info, analysis.module_name)
                if resolved:
                    import_counts[resolved] += 1
        
        most_imported = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_files": total_files,
            "total_imports": total_imports,
            "internal_imports": internal_imports,
            "external_imports": external_imports,
            "unique_external_libraries": len(external_libraries),
            "orphaned_modules": len(self.orphaned_modules),
            "circular_dependencies": len(self.circular_dependencies),
            "most_imported_modules": most_imported,
            "files_with_syntax_errors": len([f for f in self.files.values() if f.syntax_errors]),
            "external_libraries": sorted(external_libraries)
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        stats = self.get_statistics()
        
        report = []
        report.append("# Tyra MCP Memory Server - Import Dependency Analysis")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        report.append("## Summary Statistics")
        report.append(f"- Total Python files: {stats['total_files']}")
        report.append(f"- Total imports: {stats['total_imports']}")
        report.append(f"- Internal imports: {stats['internal_imports']}")
        report.append(f"- External imports: {stats['external_imports']}")
        report.append(f"- Unique external libraries: {stats['unique_external_libraries']}")
        report.append(f"- Orphaned modules: {stats['orphaned_modules']}")
        report.append(f"- Circular dependencies: {stats['circular_dependencies']}")
        report.append(f"- Files with syntax errors: {stats['files_with_syntax_errors']}")
        report.append("")
        
        # Entry points analysis
        report.append("## Entry Points Analysis")
        for entry_point in self.entry_points:
            if entry_point in self.files:
                analysis = self.files[entry_point]
                report.append(f"### {entry_point}")
                report.append(f"- Internal imports: {len(analysis.internal_imports)}")
                report.append(f"- External imports: {len(analysis.external_imports)}")
                report.append("- Key imports:")
                for import_info in analysis.internal_imports[:10]:
                    report.append(f"  - {import_info.module}")
                report.append("")
        
        # Most imported modules
        report.append("## Most Imported Internal Modules")
        for module, count in stats['most_imported_modules'][:15]:
            report.append(f"- {module}: {count} imports")
        report.append("")
        
        # Orphaned modules
        if self.orphaned_modules:
            report.append("## Orphaned Modules (Never Imported)")
            for module in sorted(self.orphaned_modules):
                report.append(f"- {module}")
            report.append("")
        
        # Circular dependencies
        if self.circular_dependencies:
            report.append("## Circular Dependencies")
            for i, cycle in enumerate(self.circular_dependencies, 1):
                report.append(f"### Cycle {i}")
                report.append(" → ".join(cycle))
                report.append("")
        
        # External libraries
        report.append("## External Libraries Used")
        for i, lib in enumerate(stats['external_libraries'], 1):
            report.append(f"{i:2d}. {lib}")
        report.append("")
        
        # Import trees from entry points
        report.append("## Import Trees from Entry Points")
        for entry_point in self.entry_points:
            if entry_point in self.files:
                analysis = self.files[entry_point]
                module_name = analysis.module_name
                report.append(f"### {entry_point} ({module_name})")
                tree = self.get_import_tree(module_name, max_depth=2)
                report.append(self._format_import_tree(tree))
                report.append("")
        
        # Files with issues
        files_with_errors = [f for f in self.files.values() if f.syntax_errors]
        if files_with_errors:
            report.append("## Files with Syntax Errors")
            for analysis in files_with_errors:
                report.append(f"### {analysis.file_path}")
                for error in analysis.syntax_errors:
                    report.append(f"- {error}")
                report.append("")
        
        # Detailed file analysis
        report.append("## Detailed File Analysis")
        
        # Sort files by directory and name
        sorted_files = sorted(self.files.items(), key=lambda x: (x[0].count('/'), x[0]))
        
        for file_path, analysis in sorted_files:
            report.append(f"### {file_path}")
            report.append(f"- Module: {analysis.module_name}")
            report.append(f"- Entry point: {analysis.is_entry_point}")
            report.append(f"- Functions: {len(analysis.functions)}")
            report.append(f"- Classes: {len(analysis.classes)}")
            report.append(f"- Total imports: {len(analysis.imports)}")
            report.append(f"- Internal imports: {len(analysis.internal_imports)}")
            report.append(f"- External imports: {len(analysis.external_imports)}")
            
            if analysis.internal_imports:
                report.append("- Internal imports:")
                for import_info in analysis.internal_imports:
                    if import_info.names:
                        report.append(f"  - from {import_info.module} import {', '.join(import_info.names)}")
                    else:
                        report.append(f"  - import {import_info.module}")
            
            if analysis.external_imports:
                report.append("- External imports:")
                for import_info in analysis.external_imports[:5]:  # Limit to first 5
                    if import_info.names:
                        report.append(f"  - from {import_info.module} import {', '.join(import_info.names)}")
                    else:
                        report.append(f"  - import {import_info.module}")
                if len(analysis.external_imports) > 5:
                    report.append(f"  - ... and {len(analysis.external_imports) - 5} more")
            
            report.append("")
        
        return "\n".join(report)
    
    def _format_import_tree(self, tree: Dict[str, Any], indent: int = 0) -> str:
        """Format import tree for display."""
        lines = []
        prefix = "  " * indent
        
        module = tree["module"]
        import_count = tree.get("import_count", 0)
        
        if indent == 0:
            lines.append(f"{prefix}📁 {module} ({import_count} imports)")
        else:
            lines.append(f"{prefix}├── {module} ({import_count} imports)")
        
        for child in tree["children"]:
            lines.append(self._format_import_tree(child, indent + 1))
        
        return "\n".join(lines)


def main():
    """Main function to run the analysis."""
    project_root = "/home/rock/ai-system/coles_ai_apps/combined_project_mcp/tyra-mcp-memory-server"
    
    analyzer = ImportAnalyzer(project_root)
    
    print("Starting comprehensive import analysis...")
    files = analyzer.analyze_all_files()
    
    print(f"Analyzed {len(files)} files")
    print(f"Found {len(analyzer.orphaned_modules)} orphaned modules")
    print(f"Found {len(analyzer.circular_dependencies)} circular dependencies")
    
    # Generate report
    report = analyzer.generate_report()
    
    # Save report to file
    with open(f"{project_root}/IMPORT_ANALYSIS_REPORT.md", "w") as f:
        f.write(report)
    
    print("\n" + "="*60)
    print("IMPORT ANALYSIS REPORT")
    print("="*60)
    print(report)
    
    # Save detailed data as JSON
    detailed_data = {
        "statistics": analyzer.get_statistics(),
        "orphaned_modules": list(analyzer.orphaned_modules),
        "circular_dependencies": analyzer.circular_dependencies,
        "dependency_graph": {k: list(v) for k, v in analyzer.dependency_graph.items()},
        "files": {
            path: {
                "module_name": analysis.module_name,
                "is_entry_point": analysis.is_entry_point,
                "imports": [
                    {
                        "module": imp.module,
                        "names": imp.names,
                        "is_from_import": imp.is_from_import,
                        "line_number": imp.line_number
                    }
                    for imp in analysis.imports
                ],
                "functions": analysis.functions,
                "classes": analysis.classes,
                "syntax_errors": analysis.syntax_errors
            }
            for path, analysis in analyzer.files.items()
        }
    }
    
    with open(f"{project_root}/import_analysis_data.json", "w") as f:
        json.dump(detailed_data, f, indent=2)
    
    print(f"\nDetailed analysis saved to:")
    print(f"- Report: {project_root}/IMPORT_ANALYSIS_REPORT.md")
    print(f"- Data: {project_root}/import_analysis_data.json")


if __name__ == "__main__":
    main()