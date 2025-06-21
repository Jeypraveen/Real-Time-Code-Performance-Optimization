import gradio as gr
import ast
import time
import tracemalloc
import sys
import io
import contextlib
import re
import google.generativeai as genai
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
import psutil
import threading
import queue
import os

# Set your Gemini API key here
GEMINI_API_KEY = ""  # Replace with your actual API key

class CodeAnalyzer:
    """Static code analysis for performance bottlenecks"""
    
    def __init__(self):
        self.issues = []
        
    def analyze_loops(self, tree):
        """Detect inefficient loops"""
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check for nested loops
                nested_loops = [n for n in ast.walk(node) if isinstance(n, ast.For) and n != node]
                if len(nested_loops) >= 2:
                    issues.append({
                        'type': 'nested_loops',
                        'line': node.lineno,
                        'severity': 'high',
                        'message': f'Deeply nested loops detected (line {node.lineno}). Consider optimization.'
                    })
                
                # Check for list operations in loops
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and hasattr(child.func, 'attr'):
                        if child.func.attr in ['append', 'extend', 'insert']:
                            issues.append({
                                'type': 'list_operations',
                                'line': node.lineno,
                                'severity': 'medium',
                                'message': f'List operations in loop (line {node.lineno}). Consider list comprehension or pre-allocation.'
                            })
        return issues
    
    def analyze_imports(self, tree):
        """Check for unused or inefficient imports"""
        issues = []
        imports = []
        used_names = set()
        
        # Collect imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.Name):
                used_names.add(node.id)
        
        # Check for unused imports (simplified)
        for imp in imports:
            if imp not in used_names:
                issues.append({
                    'type': 'unused_import',
                    'severity': 'low',
                    'message': f'Potentially unused import: {imp}'
                })
        
        return issues
    
    def analyze_functions(self, tree):
        """Analyze function complexity and performance issues"""
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Count operations in function
                operations = len([n for n in ast.walk(node) if isinstance(n, (ast.BinOp, ast.Call))])
                if operations > 50:
                    issues.append({
                        'type': 'complex_function',
                        'line': node.lineno,
                        'severity': 'medium',
                        'message': f'Function "{node.name}" is complex ({operations} operations). Consider refactoring.'
                    })
                
                # Check for recursive functions without memoization
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and hasattr(child.func, 'id'):
                        if child.func.id == node.name:
                            issues.append({
                                'type': 'recursion',
                                'line': node.lineno,
                                'severity': 'medium',
                                'message': f'Recursive function "{node.name}" detected. Consider memoization for optimization.'
                            })
        return issues

class DynamicProfiler:
    """Dynamic code profiling and performance monitoring"""
    
    def __init__(self):
        self.execution_times = []
        self.memory_usage = []
        self.cpu_usage = []
        
    def profile_code(self, code: str, test_input: str = "") -> Dict[str, Any]:
        """Profile code execution"""
        results = {
            'execution_time': 0,
            'memory_peak': 0,
            'memory_current': 0,
            'cpu_usage': 0,
            'output': '',
            'errors': [],
            'line_times': {}
        }
        
        try:
            # Start memory tracking
            tracemalloc.start()
            
            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            # Create execution environment
            exec_globals = {'__builtins__': __builtins__}
            if test_input:
                exec_globals['test_input'] = test_input
            
            # Measure execution time
            start_time = time.perf_counter()
            cpu_start = psutil.cpu_percent()
            
            # Execute code
            exec(code, exec_globals)
            
            end_time = time.perf_counter()
            cpu_end = psutil.cpu_percent()
            
            # Get results
            results['execution_time'] = end_time - start_time
            results['cpu_usage'] = (cpu_end + cpu_start) / 2
            results['output'] = captured_output.getvalue()
            
            # Memory usage
            current, peak = tracemalloc.get_traced_memory()
            results['memory_current'] = current / 1024 / 1024  # MB
            results['memory_peak'] = peak / 1024 / 1024  # MB
            
            tracemalloc.stop()
            sys.stdout = old_stdout
            
        except Exception as e:
            results['errors'].append(str(e))
            tracemalloc.stop()
            sys.stdout = old_stdout
            
        return results

class GeminiOptimizer:
    """AI-powered optimization using Gemini API"""
    
    def __init__(self, api_key: str):
        if api_key and api_key.strip():
            try:
                genai.configure(api_key=api_key)
                # Using Gemini 1.5 Flash (free tier)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.enabled = True
            except Exception as e:
                self.enabled = False
                self.error = str(e)
        else:
            self.enabled = False
            self.error = "No API key provided"
    
    def generate_optimization_suggestions(self, code: str, analysis_results: Dict) -> str:
        """Generate AI-powered optimization suggestions"""
        if not self.enabled:
            return f"Gemini API not available: {getattr(self, 'error', 'Unknown error')}"
        
        try:
            prompt = f"""
            Analyze the following Python code for performance optimization opportunities:

            CODE:
            ```python
            {code}
            ```

            ANALYSIS RESULTS:
            {json.dumps(analysis_results, indent=2)}

            Please provide specific, actionable optimization suggestions including:
            1. Code refactoring recommendations
            2. Algorithm improvements
            3. Memory optimization techniques
            4. Performance best practices
            5. Optimized code examples where applicable

            Focus on practical improvements that will have measurable performance impact.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Error generating suggestions: {str(e)}"
    
    def suggest_optimized_code(self, original_code: str, issues: List[Dict]) -> str:
        """Generate optimized version of the code"""
        if not self.enabled:
            return "Gemini API not available for code optimization"
        
        try:
            issues_summary = "\n".join([f"- {issue['message']}" for issue in issues])
            
            prompt = f"""
            Optimize the following Python code based on the identified issues:

            ORIGINAL CODE:
            ```python
            {original_code}
            ```

            IDENTIFIED ISSUES:
            {issues_summary}

            Please provide an optimized version of the code that addresses these issues while maintaining the same functionality. Include comments explaining the optimizations made.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Error generating optimized code: {str(e)}"

class PerformanceOptimizer:
    """Main optimizer class that coordinates all components"""
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.profiler = DynamicProfiler()
        self.gemini = None
        self.analysis_history = []
    
    def set_gemini_api_key(self, api_key: str):
        """Configure Gemini API"""
        self.gemini = GeminiOptimizer(api_key)
        return "✅ Gemini API configured successfully!" if self.gemini.enabled else f"❌ {self.gemini.error}"
    
    def analyze_code(self, code: str, test_input: str = "", use_ai: bool = True) -> Tuple[str, str, str]:
        """Comprehensive code analysis"""
        if not code.strip():
            return "Please enter some Python code to analyze.", "", ""
        
        try:
            # Parse code
            tree = ast.parse(code)
            
            # Static analysis
            loop_issues = self.analyzer.analyze_loops(tree)
            import_issues = self.analyzer.analyze_imports(tree)
            function_issues = self.analyzer.analyze_functions(tree)
            
            all_issues = loop_issues + import_issues + function_issues
            
            # Dynamic profiling
            profile_results = self.profiler.profile_code(code, test_input)
            
            # Create analysis report
            report = self.create_analysis_report(all_issues, profile_results)
            
            # AI suggestions
            ai_suggestions = ""
            optimized_code = ""
            
            if use_ai and self.gemini and self.gemini.enabled:
                analysis_data = {
                    'issues': all_issues,
                    'profile': profile_results,
                    'timestamp': datetime.now().isoformat()
                }
                
                ai_suggestions = self.gemini.generate_optimization_suggestions(code, analysis_data)
                if all_issues:
                    optimized_code = self.gemini.suggest_optimized_code(code, all_issues)
            elif use_ai:
                ai_suggestions = "⚠️ Gemini API not configured. Please add your API key to enable AI suggestions."
            
            # Store in history
            self.analysis_history.append({
                'timestamp': datetime.now(),
                'code_length': len(code),
                'issues_count': len(all_issues),
                'execution_time': profile_results['execution_time'],
                'memory_peak': profile_results['memory_peak']
            })
            
            return report, ai_suggestions, optimized_code
            
        except SyntaxError as e:
            return f"❌ Syntax Error: {e}", "", ""
        except Exception as e:
            return f"❌ Analysis Error: {e}", "", ""
    
    def create_analysis_report(self, issues: List[Dict], profile: Dict[str, Any]) -> str:
        """Create formatted analysis report"""
        report = "# 🔍 Code Performance Analysis Report\n\n"
        
        # Performance Metrics
        report += "## 📊 Performance Metrics\n"
        report += f"- **Execution Time**: {profile['execution_time']:.4f} seconds\n"
        report += f"- **Memory Usage**: {profile['memory_current']:.2f} MB (Peak: {profile['memory_peak']:.2f} MB)\n"
        report += f"- **CPU Usage**: {profile['cpu_usage']:.1f}%\n\n"
        
        # Issues Found
        if issues:
            report += "## ⚠️ Issues Detected\n"
            
            # Group by severity
            high_issues = [i for i in issues if i['severity'] == 'high']
            medium_issues = [i for i in issues if i['severity'] == 'medium']
            low_issues = [i for i in issues if i['severity'] == 'low']
            
            if high_issues:
                report += "### 🔴 High Priority\n"
                for issue in high_issues:
                    report += f"- {issue['message']}\n"
                report += "\n"
            
            if medium_issues:
                report += "### 🟡 Medium Priority\n"
                for issue in medium_issues:
                    report += f"- {issue['message']}\n"
                report += "\n"
            
            if low_issues:
                report += "### 🟢 Low Priority\n"
                for issue in low_issues:
                    report += f"- {issue['message']}\n"
                report += "\n"
        else:
            report += "## ✅ No Critical Issues Found\n"
            report += "Your code looks good from a performance perspective!\n\n"
        
        # Output
        if profile['output']:
            report += "## 📤 Code Output\n"
            report += f"```\n{profile['output']}\n```\n\n"
        
        # Errors
        if profile['errors']:
            report += "## ❌ Execution Errors\n"
            for error in profile['errors']:
                report += f"- {error}\n"
            report += "\n"
        
        return report
    
    def get_performance_chart(self):
        """Generate performance history chart"""
        if not self.analysis_history:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        times = [h['timestamp'] for h in self.analysis_history]
        exec_times = [h['execution_time'] for h in self.analysis_history]
        memory_usage = [h['memory_peak'] for h in self.analysis_history]
        issues_count = [h['issues_count'] for h in self.analysis_history]
        code_lengths = [h['code_length'] for h in self.analysis_history]
        
        # Execution time
        ax1.plot(times, exec_times, 'b-o')
        ax1.set_title('Execution Time')
        ax1.set_ylabel('Seconds')
        ax1.tick_params(axis='x', rotation=45)
        
        # Memory usage
        ax2.plot(times, memory_usage, 'r-o')
        ax2.set_title('Peak Memory Usage')
        ax2.set_ylabel('MB')
        ax2.tick_params(axis='x', rotation=45)
        
        # Issues count
        ax3.bar(range(len(issues_count)), issues_count, color='orange')
        ax3.set_title('Issues Detected')
        ax3.set_ylabel('Count')
        ax3.set_xlabel('Analysis #')
        
        # Code length
        ax4.bar(range(len(code_lengths)), code_lengths, color='green')
        ax4.set_title('Code Length')
        ax4.set_ylabel('Characters')
        ax4.set_xlabel('Analysis #')
        
        plt.tight_layout()
        return fig

# Initialize the optimizer with API key
optimizer = PerformanceOptimizer()
# Auto-configure Gemini API key if provided
if GEMINI_API_KEY and GEMINI_API_KEY != "your-gemini-api-key-here":
    optimizer.set_gemini_api_key(GEMINI_API_KEY)

# Gradio Interface
def analyze_code_interface(code, test_input, api_key, use_ai):
    """Interface function for Gradio"""
    # Use hardcoded API key first, then interface input, then environment variable
    api_key_to_use = None
    
    if GEMINI_API_KEY and GEMINI_API_KEY != "your-gemini-api-key-here":
        api_key_to_use = GEMINI_API_KEY
        config_msg = "✅ Using hardcoded API key"
    elif api_key and api_key.strip():
        api_key_to_use = api_key.strip()
        config_msg = optimizer.set_gemini_api_key(api_key_to_use)
    elif os.getenv('GEMINI_API_KEY'):
        api_key_to_use = os.getenv('GEMINI_API_KEY')
        config_msg = optimizer.set_gemini_api_key(api_key_to_use)
    else:
        config_msg = "❌ No API key found - AI suggestions disabled"
    
    # Configure API key if not already configured
    if api_key_to_use and not optimizer.gemini:
        optimizer.set_gemini_api_key(api_key_to_use)
    
    # Analyze code
    report, ai_suggestions, optimized_code = optimizer.analyze_code(code, test_input, use_ai)
    
    # Generate chart
    chart = optimizer.get_performance_chart()
    
    return report, ai_suggestions, optimized_code, config_msg, chart

# Example codes for demonstration
example_codes = {
    "Inefficient Loop": '''
# Example with performance issues
def find_duplicates(numbers):
    duplicates = []
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            if numbers[i] == numbers[j]:
                duplicates.append(numbers[i])
    return duplicates

# Test the function
result = find_duplicates([1, 2, 3, 2, 4, 5, 1, 6, 7, 8, 9, 1])
print("Duplicates found:", result)
''',
    
    "Memory Heavy": '''
# Memory-intensive operation
def create_large_matrix():
    matrix = []
    for i in range(1000):
        row = []
        for j in range(1000):
            row.append(i * j)
        matrix.append(row)
    return matrix

# Create and process matrix
large_matrix = create_large_matrix()
total = sum(sum(row) for row in large_matrix)
print(f"Matrix sum: {total}")
''',
    
    "Optimized Example": '''
# Well-optimized code
def find_duplicates_optimized(numbers):
    seen = set()
    duplicates = set()
    for num in numbers:
        if num in seen:
            duplicates.add(num)
        else:
            seen.add(num)
    return list(duplicates)

# Test the optimized function
result = find_duplicates_optimized([1, 2, 3, 2, 4, 5, 1, 6, 7, 8, 9, 1])
print("Duplicates found:", result)
'''
}

# Create Gradio interface
with gr.Blocks(title="AI-Powered Python Code Performance Optimizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 AI-Powered Python Code Performance Optimizer
    
    This tool provides comprehensive analysis of your Python code using both static analysis and dynamic profiling,
    enhanced with AI-powered optimization suggestions from Google's Gemini API.
    
    ## Features:
    - 🔍 **Static Analysis**: Detects inefficient loops, unused imports, and complex functions
    - 📊 **Dynamic Profiling**: Measures execution time, memory usage, and CPU utilization
    - 🤖 **AI Suggestions**: Get intelligent optimization recommendations from Gemini AI
    - 📈 **Performance Tracking**: Monitor improvements over time
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            code_input = gr.Code(
                label="Python Code to Analyze",
                language="python",
                value="# Enter your Python code here...\n",
                lines=20
            )
            
            with gr.Row():
                test_input = gr.Textbox(
                    label="Test Input (Optional)",
                    placeholder="Any test data your code might need...",
                    lines=2
                )
                
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="Gemini API Key (Optional - Leave empty if hardcoded)",
                    placeholder="API key is already configured in code...",
                    type="password"
                )
                
            with gr.Row():
                use_ai_checkbox = gr.Checkbox(
                    label="Enable AI Suggestions",
                    value=True
                )
                analyze_btn = gr.Button("🔍 Analyze Code", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📚 Example Code")
            example_dropdown = gr.Dropdown(
                choices=list(example_codes.keys()),
                label="Load Example",
                value="Inefficient Loop"
            )
            load_example_btn = gr.Button("Load Example")
    
    with gr.Row():
        api_status = gr.Textbox(
            label="API Status",
            interactive=False,
            lines=1
        )
    
    with gr.Tabs():
        with gr.TabItem("📊 Analysis Report"):
            analysis_output = gr.Markdown(label="Analysis Results")
        
        with gr.TabItem("🤖 AI Suggestions"):
            ai_output = gr.Markdown(label="AI-Powered Optimization Suggestions")
        
        with gr.TabItem("⚡ Optimized Code"):
            optimized_output = gr.Markdown(label="AI-Generated Optimized Code")
        
        with gr.TabItem("📈 Performance Chart"):
            performance_chart = gr.Plot(label="Performance History")
    
    # Event handlers
    def load_example(example_name):
        return example_codes.get(example_name, "")
    
    load_example_btn.click(
        fn=load_example,
        inputs=[example_dropdown],
        outputs=[code_input]
    )
    
    analyze_btn.click(
        fn=analyze_code_interface,
        inputs=[code_input, test_input, api_key_input, use_ai_checkbox],
        outputs=[analysis_output, ai_output, optimized_output, api_status, performance_chart]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
