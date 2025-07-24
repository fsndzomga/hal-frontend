from flask import Flask, render_template, jsonify, redirect, request
from utils.db import TracePreprocessor, DEFAULT_PRICING
from utils.viz import create_scatter_plot, create_task_success_heatmap, create_leaderboard, create_bar_chart, create_completion_tokens_bar_chart, create_model_benchmark_heatmap, create_missing_runs_heatmap
import plotly.utils
import json
from datetime import datetime

# List of contributors from creators.md
CONTRIBUTORS = [
    {"name": "Amit Arora", "affiliation": "Amazon"},
    {"name": "Aymeric Roucher", "affiliation": "Hugging Face"},
    {"name": "Hailey Schoelkopf", "affiliation": "Anthropic"},
    {"name": "Harsh Trivedi", "affiliation": "Stony Brook"},
    {"name": "Iason Gabriel", "affiliation": "Google DeepMind"},
    {"name": "Jelena Luketina", "affiliation": "UK AISI"},
    {"name": "JJ Allaire", "affiliation": "UK AISI"},
    {"name": "Laura Weidinger", "affiliation": "Google DeepMind"},
    {"name": "Madhur Prashant", "affiliation": "Amazon"},
    {"name": "Marius Hobbhahn", "affiliation": "Apollo Research"},
    {"name": "Maximillian Kaufmann", "affiliation": "UK AISI"},
    {"name": "Morgan McGuire", "affiliation": "Weights & Biases"},
    {"name": "Nitya Nadgir", "affiliation": "Princeton"},
    {"name": "Omar Khattab", "affiliation": "MIT"},
    {"name": "Parth Asawa", "affiliation": "UC Berkeley"},
    {"name": "Rishi Bommasani", "affiliation": "Stanford"},
    {"name": "Shreya Shankar", "affiliation": "UC Berkeley"},
    {"name": "Shayne Longpre", "affiliation": "MIT"},
    {"name": "Veniamin Veselovsky", "affiliation": "Princeton"},
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

        # Get models and benchmarks for the heatmap
        df = preprocessor.get_model_benchmark_accuracies()
        heatmap_fig = create_model_benchmark_heatmap(df)
        heatmap_json = json.dumps(heatmap_fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('index.html', 
                             total_agents=total_agents, 
                             total_benchmarks=total_benchmarks,
                             contributors=CONTRIBUTORS,
                             model_benchmark_heatmap=heatmap_json)
    
    @app.route("/missing")
    def missing_runs_heatmap():
        db = TracePreprocessor()
        # You may need to adjust this to get all runs with agent/model/benchmark info
        df = db.get_all_runs()  # Should return DataFrame with columns: benchmark_name, model_name, agent_name, run_id
        fig = create_missing_runs_heatmap(df)
        heatmap_json =  json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template("missing_runs_heatmap.html", heatmap_json=heatmap_json)

    @app.route('/usaco')
    def usaco():
        # Get models used in USACO benchmark
        usaco_models = preprocessor.get_models_for_benchmark('usaco')
        
        # Filter pricing to only show models used in USACO
        pricing = {model: DEFAULT_PRICING[model] for model in usaco_models if model in DEFAULT_PRICING}
        
        # Get data for USACO
        results_df = preprocessor.get_parsed_results_with_costs('usaco')
        
        # Create leaderboard
        leaderboard_df = create_leaderboard(results_df, benchmark_name='usaco')
        
        
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

        completion_tokens_fig = create_completion_tokens_bar_chart('usaco')
        completion_tokens_json = json.dumps(completion_tokens_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get last updated time
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'usaco.html',
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='usaco',  # Add benchmark name for failure analysis
            completion_tokens_bar=completion_tokens_json
        )

    @app.route('/assistantbench')
    def assistantbench():
        # Get models used in assistantbench benchmark
        assistantbench_models = preprocessor.get_models_for_benchmark('assistantbench')
        
        # Filter pricing to only show models used in assistantbench
        pricing = {model: DEFAULT_PRICING[model] for model in assistantbench_models if model in DEFAULT_PRICING}
        
        # Get data for assistantbench
        results_df = preprocessor.get_parsed_results_with_costs('assistantbench')
        
        # Create leaderboard
        leaderboard_df = create_leaderboard(results_df, benchmark_name='assistantbench')
        
        # Create scatter plot
        scatter_plot = create_scatter_plot(
            preprocessor.get_parsed_results('assistantbench', aggregate=False),
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
            preprocessor.get_task_success_data('assistantbench'),
            'AssistantBench'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)

        completion_tokens_fig = create_completion_tokens_bar_chart('assistantbench')
        completion_tokens_json = json.dumps(completion_tokens_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get last updated time
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'assistantbench.html',
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='assistantbench',  # Add benchmark name for failure analysis
            completion_tokens_bar=completion_tokens_json
        )
    
    @app.route('/update_pricing/<benchmark>', methods=['POST'])
    def update_pricing(benchmark):
        pricing = request.json
        
        if benchmark == 'agentharm':
            # Get updated results with new pricing for both benchmarks
            results_df = preprocessor.get_parsed_results_with_costs('agentharm', pricing)
            results_df_benign = preprocessor.get_parsed_results_with_costs('agentharm_benign', pricing)
            
            # Create updated leaderboards
            leaderboard_df = create_leaderboard(results_df, benchmark_name='agentharm')
            leaderboard_df_benign = create_leaderboard(results_df_benign, benchmark_name='agentharm_benign')
            
            # Create updated scatter plots
            scatter_plot = create_scatter_plot(
                results_df,
                "Total Cost",
                "Accuracy",
                "Total Cost (in USD)",
                "Accuracy",
                ["Agent Name"]
            )
            scatter_plot_benign = create_scatter_plot(
                results_df_benign,
                "Total Cost",
                "Accuracy",
                "Total Cost (in USD)",
                "Accuracy",
                ["Agent Name"]
            )
            
            # Convert scatter plots to JSON
            scatter_plot_json = json.loads(json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder))
            scatter_plot_benign_json = json.loads(json.dumps(scatter_plot_benign, cls=plotly.utils.PlotlyJSONEncoder))
            
            return jsonify({
                'leaderboard': leaderboard_df.to_dict('records'),
                'leaderboard_benign': leaderboard_df_benign.to_dict('records'),
                'scatter_plot': scatter_plot_json,
                'scatter_plot_benign': scatter_plot_benign_json
            })
        else:
            # Get updated results with new pricing
            results_df = preprocessor.get_parsed_results_with_costs(benchmark, pricing)
                        
            # Create updated leaderboard
            leaderboard_df = create_leaderboard(results_df, benchmark_name=benchmark)
            
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

    @app.route('/corebench_hard')
    def corebench_hard():
        corebench_models = preprocessor.get_models_for_benchmark('corebench_hard')
        pricing = {model: DEFAULT_PRICING[model] for model in corebench_models if model in DEFAULT_PRICING}
        
        results_df = preprocessor.get_parsed_results_with_costs('corebench_hard')
        leaderboard_df = create_leaderboard(results_df, benchmark_name='corebench_hard')
        
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('corebench_hard'),
            'CORE-Bench-Hard'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)

        completion_tokens_fig = create_completion_tokens_bar_chart('corebench_hard')
        completion_tokens_json = json.dumps(completion_tokens_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'corebench.html',
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            difficulty="Hard",
            benchmark_name='corebench_hard',  # Add benchmark name for failure analysis
            completion_tokens_bar=completion_tokens_json
        )

    @app.route('/gaia')
    def gaia():
        gaia_models = preprocessor.get_models_for_benchmark('gaia')
        pricing = {model: DEFAULT_PRICING[model] for model in gaia_models if model in DEFAULT_PRICING}
        
        results_df = preprocessor.get_parsed_results_with_costs('gaia')
        leaderboard_df = create_leaderboard(results_df, benchmark_name='gaia')
                
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('gaia'),
            'GAIA'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)

        completion_tokens_fig = create_completion_tokens_bar_chart('gaia')
        completion_tokens_json = json.dumps(completion_tokens_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'gaia.html',  # Will need to create this template
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='gaia',  # Add benchmark name for failure analysis
            completion_tokens_bar=completion_tokens_json
        )

    @app.route('/taubench_airline')
    def taubench_airline():
        # Get models used in TAU-bench Airline benchmark
        taubench_airline_models = preprocessor.get_models_for_benchmark('taubench_airline')
        pricing = {model: DEFAULT_PRICING[model] for model in taubench_airline_models if model in DEFAULT_PRICING}
        
        # Get data for TAU-bench Airline
        results_df = preprocessor.get_parsed_results_with_costs('taubench_airline')
        
        # Create leaderboard
        leaderboard_df = create_leaderboard(results_df, benchmark_name='taubench_airline')
        
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
            preprocessor.get_task_success_data('taubench_airline'),
            'TAU-bench Airline'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)

        completion_tokens_fig = create_completion_tokens_bar_chart('taubench_airline')
        completion_tokens_json = json.dumps(completion_tokens_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get last updated time
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'taubench_airline.html',
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='taubench_airline',
            completion_tokens_bar=completion_tokens_json
        )

    @app.route('/swebench_verified_mini')
    def swebench_verified_mini():
        swebench_models = preprocessor.get_models_for_benchmark('swebench_verified_mini')
        pricing = {model: DEFAULT_PRICING[model] for model in swebench_models if model in DEFAULT_PRICING}
        
        results_df = preprocessor.get_parsed_results_with_costs('swebench_verified_mini')
        leaderboard_df = create_leaderboard(results_df, benchmark_name='swebench_verified_mini')
        
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('swebench_verified_mini'),
            'SWE-bench Verified (Mini)'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)

        completion_tokens_fig = create_completion_tokens_bar_chart('swebench_verified_mini')
        completion_tokens_json = json.dumps(completion_tokens_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'swebench_verified_mini.html',  # Use the new template
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='swebench_verified_mini',  # Add benchmark name for failure analysis
            completion_tokens_bar=completion_tokens_json
        )
    
    @app.route('/online_mind2web')
    def online_mind2web():
        online_mind2web_models = preprocessor.get_models_for_benchmark('online_mind2web')
        pricing = {model: DEFAULT_PRICING[model] for model in online_mind2web_models if model in DEFAULT_PRICING}
        
        results_df = preprocessor.get_parsed_results_with_costs('online_mind2web')
        leaderboard_df = create_leaderboard(results_df, benchmark_name='online_mind2web')

        completion_tokens_fig = create_completion_tokens_bar_chart('online_mind2web')
        completion_tokens_json = json.dumps(completion_tokens_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('online_mind2web'),
            'Online Mind2Web'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)
        
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'online_mind2web.html',  # Use the new template
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='online_mind2web',  # Add benchmark name for failure analysis
            completion_tokens_bar=completion_tokens_json
        )
    
    @app.route('/scicode')
    def scicode():
        scicode_models = preprocessor.get_models_for_benchmark('scicode')
        pricing = {model: DEFAULT_PRICING[model] for model in scicode_models if model in DEFAULT_PRICING}
        
        results_df = preprocessor.get_parsed_results_with_costs('scicode')
        leaderboard_df = create_leaderboard(results_df, benchmark_name='scicode')
        
        scatter_plot = create_scatter_plot(
            results_df,
            "Total Cost",
            "Accuracy",
            "Total Cost (in USD)",
            "Accuracy",
            ["Agent Name"]
        )
        scatter_plot_json = json.dumps(scatter_plot, cls=plotly.utils.PlotlyJSONEncoder)
        
        heatmap = create_task_success_heatmap(
            preprocessor.get_task_success_data('scicode'),
            'Scicode'
        )
        heatmap_json = json.dumps(heatmap, cls=plotly.utils.PlotlyJSONEncoder)

        completion_tokens_fig = create_completion_tokens_bar_chart('scicode')
        completion_tokens_json = json.dumps(completion_tokens_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        return render_template(
            'scicode.html',  # Use the new template
            leaderboard=leaderboard_df.to_dict('records'),
            scatter_plot=scatter_plot_json,
            heatmap=heatmap_json,
            last_updated=last_updated,
            pricing=pricing,
            benchmark_name='scicode',  # Add benchmark name for failure analysis
            completion_tokens_bar=completion_tokens_json
        )


    @app.route('/failure_report/<benchmark>')
    def failure_report(benchmark):
        agent_name = request.args.get('agent')
        if not agent_name:
            return jsonify({'error': 'Agent name is required'}), 400
            
        # Get failure report for the agent
        failure_report = preprocessor.get_failure_report(agent_name, benchmark)
        if not failure_report:
            return jsonify({
                'failure_categories': [],
                'chart_data': None
            })
            
        # Create bar chart data
        categories = []
        counts = []
        for category in failure_report['failure_categories']:
            categories.append(category['category_name'])
            # Count tasks in this category
            count = sum(1 for _, classification in failure_report['task_classifications'].items() 
                       if classification['category_id'] == str(len(categories)))
            counts.append(count)
            
        chart = create_bar_chart(
            categories,
            counts,
            "Number of Tasks",
            "Failure Categories",
            "Distribution of Failure Categories"
        )
        
        return jsonify({
            'failure_categories': failure_report['failure_categories'],
            'chart_data': json.loads(json.dumps(chart, cls=plotly.utils.PlotlyJSONEncoder))
        })

    @app.route('/available_agents/<benchmark>')
    def available_agents(benchmark):
        # Get all agents for the benchmark
        agents = preprocessor.get_all_agents(benchmark)
        
        # Filter to only agents with failure reports
        agents_with_reports = [
            agent for agent in agents 
            if preprocessor.get_failure_report(agent, benchmark) is not None
        ]
        
        return jsonify(agents_with_reports)

    @app.route('/creators')
    def creators():
        return render_template('creators.html', contributors=CONTRIBUTORS)

    return app


app = create_app()

if __name__ == '__main__':
    app.run(debug=False, port=5001, host='0.0.0.0')