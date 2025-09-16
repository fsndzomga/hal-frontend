# LaTeX Benchmark Sections

This folder contains LaTeX sections for all HAL benchmark results. Each section follows the same structure with benchmark description, agent placeholder, leaderboard table, and 4 figures (3 for ScienceAgentBench).

## Files Generated

### Individual Benchmark Sections:
- `assistantbench.tex` - AssistantBench (214 tasks, web browsing)
- `gaia.tex` - GAIA (450 questions, general AI assistant tasks)
- `corebench.tex` - CORE-Bench (scientific paper reproduction)
- `swebench.tex` - SWE-bench Verified Mini (50 GitHub issues)
- `mind2web.tex` - Online Mind2Web (live web interaction)
- `scicode.tex` - SciCode (338 scientific coding subproblems)
- `scienceagentbench.tex` - ScienceAgentBench (102 data-driven discovery tasks) *
- `taubench_airline.tex` - TAU-bench Airline (airline domain tasks)
- `usaco.tex` - USACO (307 competitive programming problems)

### Helper Files:
- `all_benchmarks.tex` - Master file including all sections
- `unicode_setup.tex` - Unicode character definitions for LaTeX
- `generate_latex_tables.py` - Script to generate tables from leaderboard data
- `README.md` - Comprehensive documentation and usage guide

### Generated Files:
- `tables/` - Directory containing all LaTeX table files (auto-generated)

## Figure References

Each benchmark section references these figures (replace `{benchmark_abbrev}` with actual prefix):

1. `{abbrev}_pareto_accuracy_vs_cost.pdf` - Pareto frontier scatter plot
2. `{abbrev}_total_tokens.pdf` - Token usage bar chart  
3. `{abbrev}_heatmap_best_vs_any.pdf` - Success rate heatmap
4. `{abbrev}_accuracy_vs_release_date.pdf` - Timeline chart

### Benchmark Abbreviations:
- `ab` - AssistantBench
- `gaia` - GAIA
- `core` - CORE-Bench  
- `swebench` - SWE-bench Verified Mini
- `mind2web` - Online Mind2Web
- `scicode` - SciCode
- `sab` - ScienceAgentBench
- `tau_airline` - TAU-bench Airline
- `usaco` - USACO

## Table References

Each section references a table file in `tables/{benchmark}_full_results.tex`. These tables are automatically generated from the leaderboard data using the `generate_latex_tables.py` script.

## Usage

1. Copy the `.tex` files to your LaTeX document folder
2. Copy the corresponding PDF figures to your figures folder
3. Copy the `tables/` directory with all generated table files
4. Include `unicode_setup.tex` in your document preamble for Unicode support
5. Include individual sections or use `\input{all_benchmarks.tex}`
6. Update the "Agents" paragraph with your specific agent descriptions

## Generating Tables

To regenerate the LaTeX tables from fresh data:

```bash
python generate_latex_tables.py
```

This will create clean, LaTeX-compatible table files from the current leaderboard data.

## Notes

- ScienceAgentBench (*) only has 3 figures (no heatmap) as the heatmap is disabled for this benchmark
- All citations use placeholder labels (e.g., `\cite{assistantbench}`) - update with your bibliography
- Placeholder text "Briefly describe the two agents..." needs to be replaced with actual agent descriptions
- Table labels and file paths may need adjustment based on your document structure

## Required LaTeX Packages

```latex
\usepackage{adjustbox}  % for figure sizing
\usepackage{graphicx}   % for includegraphics
```