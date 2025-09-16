#!/usr/bin/env python3
"""
Test script to verify dynamic cost cutoff behavior
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from utils.db import TracePreprocessor
from utils.viz import create_scatter_plot

def test_cutoff_logic():
    """Test if cutoff_applied flag is set correctly for different benchmarks"""
    
    preprocessor = TracePreprocessor()
    
    # Test different benchmarks
    benchmarks = ['assistantbench', 'usaco', 'gaia', 'corebench_hard']
    
    for benchmark in benchmarks:
        try:
            print(f"\n=== Testing {benchmark.upper()} ===")
            
            # Get data
            results_df = preprocessor.get_parsed_results_with_costs(benchmark)
            if results_df.empty:
                print(f"No data found for {benchmark}")
                continue
                
            # Test with cutoff
            scatter_plot, cutoff_applied = create_scatter_plot(
                results_df,
                "Total Cost",
                "Accuracy", 
                "Total Cost (in USD)",
                "Accuracy",
                ["Agent Name"],
                cost_cutoff_multiplier=10.0,
                return_cutoff_applied=True
            )
            
            # Show cost statistics
            costs = results_df.groupby('Agent Name')['Total Cost'].mean()
            max_accuracy = results_df.groupby('Agent Name')['Accuracy'].mean().max()
            most_accurate_agents = results_df.groupby('Agent Name')['Accuracy'].mean()
            most_accurate_cost = most_accurate_agents[most_accurate_agents == max_accuracy].index[0]
            most_accurate_cost_value = costs[most_accurate_cost]
            
            cutoff_threshold = most_accurate_cost_value * 10.0
            max_cost = costs.max()
            
            print(f"Most accurate agent cost: ${most_accurate_cost_value:.2f}")
            print(f"10x cutoff threshold: ${cutoff_threshold:.2f}")
            print(f"Maximum agent cost: ${max_cost:.2f}")
            print(f"Agents beyond threshold: {sum(costs > cutoff_threshold)}")
            print(f"Cutoff applied: {cutoff_applied}")
            print(f"Logic check: max_cost > threshold = {max_cost > cutoff_threshold}")
            
        except Exception as e:
            print(f"Error testing {benchmark}: {e}")

if __name__ == "__main__":
    test_cutoff_logic()