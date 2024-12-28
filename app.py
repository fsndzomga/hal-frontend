from flask import Flask, render_template, jsonify, redirect, request
from utils.db import TracePreprocessor, DEFAULT_PRICING
from utils.viz import create_scatter_plot, create_task_success_heatmap, create_leaderboard
import plotly.utils
import json
from datetime import datetime

# List of contributors from creators.md
CONTRIBUTORS = [
    {"name": "Amit Arora", "affiliation": "Amazon"},
    {"name": "Aymeric Roucher", "affiliation": "Hugging Face"},
    {"name": "Azalia Mirhoseini", "affiliation": "Stanford"},
    {"name": "Bagatur Askaryan", "affiliation": "LangChain"},
    {"name": "Hailey Schoelkopf", "affiliation": "Anthropic"},
    {"name": "Harsh Trivedi", "affiliation": "Stony Brook"},
    {"name": "Iason Gabriel", "affiliation": "Google DeepMind"},
    {"name": "Jelena Luketina", "affiliation": "UK AISI"},
    {"name": "JJ Allaire", "affiliation": "UK AISI"},
    {"name": "Karthik Narasimhan", "affiliation": "Princeton"},
    {"name": "Kwamina Orleans-Pobee", "affiliation": "DSIT UK"},
    {"name": "Laura Weidinger", "affiliation": "Google DeepMind"},
    {"name": "Madhur Prashant", "affiliation": "Amazon"},
    {"name": "Maximillian Kaufmann", "affiliation": "UK AISI"},
    {"name": "Morgan McGuire", "affiliation": "Weights & Biases"},
    {"name": "Nitya Nadgir", "affiliation": "Princeton"},
    {"name": "Parth Asawa", "affiliation": "UC Berkeley"},
    {"name": "Rishi Bommasani", "affiliation": "Stanford"},
    {"name": "Shreya Shankar", "affiliation": "UC Berkeley"},
    {"name": "Shayne Longpre", "affiliation": "MIT"},
    {"name": "Thomas Capelle", "affiliation": "Weights & Biases"},
    {"name": "William Isaac", "affiliation": "Google DeepMind"},
    {"name": "Yifan Mai", "affiliation": "Stanford"},
    {"name": "Zachary Siegel", "affiliation": "Princeton"}
]

def create_app():
    app = Flask(__name__, 
               static_folder='static',
               template_folder='templates')
    
    preprocessor = TracePreprocessor()
    
    @app.route('/')
    def index():
        total_agents = preprocessor.get_total_agents()
        total_benchmarks = preprocessor.get_total_benchmarks()
        return render_template('index.html', 
                             total_agents=total_agents, 
                             total_benchmarks=total_benchmarks)

    @app.route('/swebench')
    def swebench():
        # Get models used in SWE-bench benchmark
        swebench_models = preprocessor.get_models_for_benchmark('swebench_verified')
        
        # Filter pricing to only show models used in SWE-bench
        pricing = {model: DEFAULT_PRICING[model] for model in swebench_models if model in DEFAULT_PRICING}
        
        # Get data for SWE-bench
        results_df = preprocessor.get_parsed_results_with_costs('swebench_verified')
        print(results_df)
        # Create leaderboard
        leaderboard_df = create_leaderboard(results_df, ci_metrics=["Accuracy", "Total Cost"])
        # Create scatter plot
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        
        # Convert plot to JSON for rendering
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create heatmap
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('swebench_verified'),
            'SWE-bench Verified'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get last updated time
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'swebench.html',
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing
        )

    @app.route('/usaco')
    def usaco():
        # Get models used in USACO benchmark
        usaco_models = preprocessor.get_models_for_benchmark('usaco')
        
        # Filter pricing to only show models used in USACO
        pricing = {model: DEFAULT_PRICING[model] for model in usaco_models if model in DEFAULT_PRICING}
        
        # Get data for USACO
        results_df = preprocessor.get_parsed_results_with_costs('usaco')
        
        # Create leaderboard
        leaderboard_df = create_leaderboard(results_df, ci_metrics=["Accuracy", "Total Cost"])
        
        # Create scatter plot
        scatter_plot = create_scatter_plot(
            preprocessor.get_parsed_results('usaco', aggregate=False),
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        
        # Convert plot to JSON for rendering
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create heatmap
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('usaco'),
            'USACO'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get last updated time
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'usaco.html',
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing
        )

    @app.route('/update_pricing/<benchmark>', methods=['POST'])
    def update_pricing(benchmark):
        pricing = request.json
        
        # Get updated results with new pricing
        results_df = preprocessor.get_parsed_results_with_costs(benchmark, pricing)
        
        # Create updated leaderboard
        leaderboard_df = create_leaderboard(results_df, ci_metrics=["Accuracy", "Total Cost"])
        
        # Create updated scatter plot
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        
        # Convert scatter plot to JSON
        scatter_plot_json = json.loads(json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder))
        
        return jsonify({
            'leaderboard': leaderboard_df.to_dict('records'),
            'scatter_plot': scatter_plot_json
        })

    @app.route('/appworld')
    def appworld():
        # Get models used in AppWorld benchmark
        appworld_models = preprocessor.get_models_for_benchmark('appworld_test_normal')
        
        # Filter pricing to only show models used in AppWorld
        pricing = {model: DEFAULT_PRICING[model] for model in appworld_models if model in DEFAULT_PRICING}
        
        # Get data for AppWorld
        results_df = preprocessor.get_parsed_results_with_costs('appworld_test_normal')
        
        # Create leaderboard
        leaderboard_df = create_leaderboard(results_df, ci_metrics=["Accuracy", "Total Cost"])
        
        # Create scatter plot
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        
        # Convert plot to JSON for rendering
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create heatmap
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('appworld_test_normal'),
            'AppWorld'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get last updated time
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'appworld.html',
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing
        )

    @app.route('/creators')
    def creators():
        return render_template('creators.html', contributors=CONTRIBUTORS)

    return app


app = create_app()

if __name__ == '__main__':
    app.run(debug=False, port=5001, host='0.0.0.0')