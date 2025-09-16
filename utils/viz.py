import json
import plotly.express as px
from utils.pareto import Agent, compute_pareto_frontier
import plotly.graph_objects as go
import textwrap
import numpy as np
import pandas as pd
from scipy import stats
from plotly.subplots import make_subplots
from utils.db import DEFAULT_PRICING, TracePreprocessor
from utils.pareto import Agent, compute_pareto_frontier

def create_missing_runs_heatmap(df):
    df['agent_name'] = (
        df['agent_name']
        .str.replace(r'\s*\(.*\)\s*$', '', regex=True)
        .str.strip()
    )

    # remove some rows based on a list of model names
    models_to_remove = ["GPT-OSS-120B", "GPT-OSS-120B High", "Claude Opus 4 (May 2025)", 
                        "Claude Opus 4 High (May 2025)", "Claude Sonnet 4 (May 2025)",
                        "Claude Sonnet 4 High (May 2025)", "Gemini 2.5 Pro Preview (March 2025)"]
    
    df = df[~df['model_name'].isin(models_to_remove)]

    # save df to a CSV file for debugging
    df.to_csv('missing_runs_heatmap_debug.csv', index=False)
    
    # Combine benchmark and agent for unique columns, ordered alphabetically by both
    df['bench_agent'] = df['benchmark_name'] + " | " + df['agent_name']

    # remove some rows based on bench_agent list
    bench_agents_to_remove = ["assistantbench | HAL Generalist Agent",
                              "gaia | HAL Generalist Agent",
                              "usaco | HAL Generalist Agent",
                              "scicode | HAL Generalist Agent",
                              "scicode | Scicode Zero Shot Agent"
                              ]
    
    df = df[~df['bench_agent'].isin(bench_agents_to_remove)]

    pivot = df.pivot_table(
        index='model_name', 
        columns='bench_agent', 
        values='run_id', 
        aggfunc='count'
    )

    # Sort columns alphabetically
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    
    missing = pivot.isna().astype(int)
    text = np.where(missing.values, "Missing", "Present")

    fig = go.Figure(data=go.Heatmap(
        z=missing.values,
        x=pivot.columns,
        y=pivot.index,
        text=text,
        texttemplate="%{text}",
        colorscale=[[1, "#f8d7da"], [0, "#d4edda"]],
        showscale=False,
        hovertemplate="<b>Model:</b> %{y}<br><b>Benchmark | Agent:</b> %{x}<br><b>Status:</b> %{text}<extra></extra>",
        xgap=2, ygap=2
    ))

    fig.update_layout(
        title="Missing Runs Heatmap",
        margin=dict(l=180, r=40, t=60, b=180),  # Increased bottom margin for labels
        xaxis=dict(
            title="Benchmark | Agent",
            tickangle=-45,
            tickfont=dict(size=10),  # Smaller font size for better fit
            automargin=True,  # Enable auto-margin
        ),
        yaxis=dict(
            title="Model Name", 
            automargin=True
        ),
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig

def create_model_benchmark_heatmap(df):
    # Pivot to matrix
    pivot = df.pivot(index='benchmark_name', columns='model_name', values='accuracy')
    pivot = pivot.sort_index().sort_index(axis=1)
    z = pivot.values

    # Prepare text annotations
    text = np.empty(z.shape, dtype=object)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if np.isnan(z[i, j]):
                text[i, j] = "no runs"
            else:
                text[i, j] = f"{z[i, j]*100:.0f}%"

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=pivot.columns,
        y=pivot.index,
        text=text,
        texttemplate="%{text}",
        colorscale=[
            [0.0, "#e3f2fd"],   # very light blue
            [0.2, "#90caf9"],   # lighter blue
            [0.4, "#64b5f6"],   # light blue
            [0.6, "#42a5f5"],   # medium blue
            [0.8, "#2880B9"],   # main blue
            [1.0, "#1565c0"]    # dark blue
        ],
        zmin=0,
        zmax=0.8,
        colorbar=dict(title="Accuracy", tickformat=".0%"),
        hovertemplate="<b>Benchmark:</b> %{y}<br><b>Model:</b> %{x}<br><b>Accuracy:</b> %{text}<extra></extra>",
        showscale=True,
        hoverongaps=False,
        xgap=2, ygap=2
    ))

    # Add annotation for "no runs" cells (dark grey text)
    annotations = []
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if np.isnan(z[i, j]):
                annotations.append(dict(
                    x=pivot.columns[j],
                    y=pivot.index[i],
                    text="no runs",
                    font=dict(color="#444", size=11, family="Arial"),
                    showarrow=False
                ))

    fig.update_layout(
        margin=dict(l=180, r=40, t=40, b=80),
        xaxis=dict(title="Model Name", tickangle=-55, side='bottom'),
        yaxis=dict(title="Benchmark Name", automargin=True, tickfont=dict(size=13, family="Arial", color="black")),
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white',
        annotations=annotations
    )
    return fig

def create_completion_tokens_bar_chart(benchmark_name, pricing_config=None, top_n=20):
    db = TracePreprocessor()
    df = db.get_token_usage_with_costs(benchmark_name, pricing_config)
    if df.empty:
        return go.Figure()

    # keep only the primary model rows, if that column exists
    if 'is_primary' in df.columns:
        df = df[df.get('is_primary', 1) == 1]

    # Sum completion tokens within each run first
    run_totals = (df
        .groupby(['agent_name', 'model_name', 'run_id'], as_index=False)
        ['completion_tokens'].sum())

    #Sum over all runs and all models for the agent
    agent_totals = (run_totals
        .groupby('agent_name', as_index=False)
        ['completion_tokens'].sum()
        .sort_values('completion_tokens', ascending=False)
        .head(top_n))

    # Convert to millions and build the bar
    agent_totals['completion_tokens_m'] = agent_totals['completion_tokens'] / 1e6
    agent_totals['label'] = agent_totals['agent_name']   # one bar per agent

    fig = go.Figure(go.Bar(
        y=agent_totals['label'],
        x=agent_totals['completion_tokens_m'],
        orientation='h',
        marker_color='#3498db',
        text=[f"{v:.2f} M" for v in agent_totals['completion_tokens_m']],
        textposition='auto',
        hovertemplate="<b>%{y}</b><br>Total completion tokens: %{x:.2f} M<extra></extra>"
    ))

    fig.update_layout(
        xaxis_title="Completion Tokens Used (Millions)",
        yaxis_title="Agent",
        height=max(400, 40 * len(agent_totals)),
        margin=dict(l=200, r=40, t=60, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

def create_leaderboard(df, benchmark_name = None):
    # Define the columns we want to aggregate and their aggregation methods
    desired_agg = {
        'Date': 'first',
        'Total Cost': 'mean',
        'Accuracy': 'mean',
        'Precision': 'mean',
        'Recall': 'mean',
        'F1 Score': 'mean',
        'AUC': 'mean',
        'Overall Score': 'mean',
        'Vectorization Score': 'mean',
        'Fathomnet Score': 'mean',
        'Feedback Score': 'mean',
        'House Price Score': 'mean',
        'Spaceship Titanic Score': 'mean',
        'AMP Parkinsons Disease Progression Prediction Score': 'mean',
        'CIFAR10 Score': 'mean',
        'IMDB Score': 'mean',
        'Scenario Goal Completion': 'mean',
        'Task Goal Completion': 'mean',
        'Level 1 Accuracy': 'mean',
        'Level 2 Accuracy': 'mean',
        'Level 3 Accuracy': 'mean',
        'Verified': 'first',
        'Traces': 'first',
        'Runs': 'first',
        'Accuracy CI': 'first',
        'Total Cost CI': 'first',
        'URL': 'first',
        'Refusals': 'mean',
        'Non-Refusal Harm Score': 'mean',
        'Model Name': 'first',
    }
    
    # Only use columns that actually exist in the DataFrame
    actual_agg = {col: agg for col, agg in desired_agg.items() if col in df.columns}
    
    df = df.groupby('Agent Name', as_index=False).agg(actual_agg)

    # Sort by Accuracy (AgentHarm ascending, others descending)
    if benchmark_name == 'agentharm':
        df = df.sort_values('Accuracy', ascending=True)
    else:
        # In case of ties in Accuracy, sort by Total Cost ascending
        df = df.sort_values(['Accuracy', 'Total Cost'], ascending=[False, True])

    # Compute Pareto frontier on numeric columns (using agent means like scatter plot)
    df['Total Cost'] = pd.to_numeric(df['Total Cost'], errors='coerce')
    df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')

    # Get unique agents and their mean costs/accuracy (consistent with scatter plot)
    unique_agents = df['Agent Name'].unique()
    agent_means = {
        agent: (
            df[df['Agent Name'] == agent]['Total Cost'].mean(),
            df[df['Agent Name'] == agent]['Accuracy'].mean()
        )
        for agent in unique_agents
    }
    
    # Filter out agents with NaN values
    valid_agent_means = {agent: (cost, acc) for agent, (cost, acc) in agent_means.items() 
                        if pd.notna(cost) and pd.notna(acc)}
    
    if len(valid_agent_means) > 0:
        agents = [Agent(cost, acc) for cost, acc in valid_agent_means.values()]
        frontier = compute_pareto_frontier(agents)
        frontier_pts = {(round(a.total_cost, 6), round(a.accuracy, 6)) for a in frontier}
        
        # Mark agents as Pareto optimal based on their mean values
        df['Is Pareto'] = df.apply(
            lambda r: (round(agent_means[r['Agent Name']][0], 6), round(agent_means[r['Agent Name']][1], 6)) in frontier_pts
            if r['Agent Name'] in valid_agent_means else False, axis=1
        )
    else:
        df['Is Pareto'] = False

    # Count runs (kept for compatibility)
    runs_per_agent = df.groupby('Agent Name').size().reset_index()
    runs_per_agent.columns = ['Agent Name', 'Runs']
    runs_per_agent.drop("Runs", inplace=True, axis=1)
    df = df.merge(runs_per_agent, on='Agent Name', how='left')

    # Add model names to leaderboard data
    df.rename(columns={'Model Name': 'Models'}, inplace=True)

    # Remove model names from agent name
    df['Agent Name'] = (
        df['Agent Name']
        .str.replace(r'\s*\(.*\)\s*$', '', regex=True)
        .str.strip()
    )
    return df

def create_task_success_heatmap(df, benchmark_name):
    
    if 'AgentHarm' in benchmark_name:
        df['Task ID'] = df['Task ID'].str.replace('-', '_') # TODO - remove hardcoding
    
    # Calculate agent accuracy (now using mean success rate)
    agent_accuracy = df.groupby('Agent Name')['Success'].mean().sort_values(ascending=False)
    
    # Calculate task success rate (first take mean success rate by task and agent, then take mean success rate by task)
    task_success_rate = df.groupby(['Task ID', 'Agent Name'])['Success'].mean().unstack().mean(axis=1).sort_values(ascending=False)
    
    # Pivot the dataframe to create a matrix of agents vs tasks
    pivot_df = df.pivot(index='Agent Name', columns='Task ID', values='Success')
    
    # Sort the pivot table
    pivot_df = pivot_df.reindex(index=agent_accuracy.index, columns=task_success_rate.index)

    # Calculate tasks solved across all agents (considering a task solved if any agent solved it in any run)
    tasks_solved = (pivot_df.max(axis=0) > 0).astype(int)
    # Total number of tasks (columns)
    total_tasks = len(pivot_df.columns)
    
    if benchmark_name == "SWE-bench Verified (Mini)":
        total_tasks = 50 # TODO - remove hardcoding

    # Calculate best agent performance per task
    best_agent_performance = pivot_df.max(axis=0)
    
    # Calculate the difference between best agent and tasks solved
    performance_gap = best_agent_performance - tasks_solved
    
    # Add the tasks solved row to the pivot table
    tasks_solved_df = pd.DataFrame(tasks_solved).T
    tasks_solved_df.index = [f'<b>Tasks Solved: {tasks_solved.sum()}/{total_tasks} (Any Agent)</b>']
    
    # Combine rows for the heatmap
    pivot_df = pd.concat([pivot_df, tasks_solved_df])

    num_agents = len(pivot_df.index)
    row_height = 30  # Fixed height for each row in pixels
    total_height = num_agents * row_height
    
    # Create a custom colorscale that goes from white to blue
    colorscale = [
        [0, 'white'],
        [0.01, '#EBF5FB'],  # Very light blue
        [0.25, '#BFE0F5'],  # Lighter blue
        [0.5, '#93CAF1'],   # Light blue
        [0.75, '#67B4EC'],  # Medium blue
        [1, '#3498db']      # Target blue (#3498db)
    ]

    # Create figure with subplots - one for heatmap, one for summary
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.8, 0.2],
        vertical_spacing=0.1,
        subplot_titles=('Task ID', 'Performance Gap')
    )
    
    # Add heatmap to first subplot
    fig.add_trace(
        go.Heatmap(
            z=pivot_df.values,
            y=pivot_df.index,
            x=pivot_df.columns,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title="Fraction of Runs",
                tickformat=".0%"
            ),
            hovertemplate='<b>Agent:</b> %{y}<br>' +
                         '<b>Task:</b> %{x}<br>' +
                         '<b>Fraction of Runs:</b> %{z:.1%}<extra></extra>'
        ),
        row=1, col=1
    )

    # Calculate summary statistics
    tasks_solved_count = tasks_solved.sum()
    best_agent_accuracy = agent_accuracy.iloc[0]  # Get the best agent's overall success rate
    best_agent_solved = int(best_agent_accuracy * total_tasks)  # Convert to number of tasks
    
    # Add bar chart for summary
    fig.add_trace(
        go.Bar(
            x=[tasks_solved_count, best_agent_solved],
            y=['Any agent', 'Best agent'],
            orientation='h',
            text=[f'{tasks_solved_count}/{total_tasks} ({tasks_solved_count/total_tasks:.1%})',
                  f'{best_agent_solved}/{total_tasks} ({best_agent_accuracy:.1%})'],
            textposition='auto',
            marker_color=['#3498db', '#2ecc71'],
            showlegend=False,
            hovertemplate='%{text}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update the layout
    fig.update_layout(
        height=total_height + 150,  # Add extra space for the summary
        plot_bgcolor='white',
        paper_bgcolor='white',
        hoverlabel=dict(
            bgcolor="white", 
            font_size=12, 
            font_family="Arial"
        ),
        modebar=dict(
            activecolor='#1f77b4',
            orientation='h',
            bgcolor='rgba(255,255,255,0.8)',
            color='#777',
            add=['pan2d'],
            remove=[
                'zoom2d', 'zoomIn2d', 'zoomOut2d', 'resetScale2d',
                'hoverClosestCartesian', 'hoverCompareCartesian',
                'toggleSpikelines', 'lasso2d', 'lasso', 'select2d', 'select'
            ]
        ),
        dragmode='pan',
        margin=dict(t=50, l=150, r=50, b=20),  # Increased left margin for bar labels
        showlegend=False
    )
    
    # Update yaxis properties for heatmap
    fig.update_yaxes(
        autorange='reversed',
        showticklabels=True,
        showline=True,
        linecolor='black',
        showgrid=False,
        row=1, col=1
    )
    
    # Update xaxis properties for heatmap
    fig.update_xaxes(
        title='',
        side='top',
        showticklabels=False,
        showline=True,
        linecolor='black',
        showgrid=False,
        row=1, col=1
    )
    
    # Update axes properties for bar chart
    fig.update_yaxes(
        showticklabels=True,
        showline=True,
        linecolor='black',
        showgrid=False,
        row=2, col=1
    )
    
    fig.update_xaxes(
        range=[0, total_tasks],
        showticklabels=True,
        showline=True,
        linecolor='black',
        showgrid=False,
        row=2, col=1
    )

    return fig

def create_bar_chart(categories, values, x_label, y_label, title):
    # Sort categories and values based on values in descending order
    sorted_data = sorted(zip(categories, values), key=lambda x: x[1], reverse=True)
    categories, values = zip(*sorted_data)

    # get total number of tasks
    total_tasks = sum(values)

    text_labels = [f"({value/total_tasks:.1%} of failures)" for value in values]

    fig = go.Figure(data=[go.Bar(
        y=categories,
        x=values,
        orientation='h',
        marker_color='#3498db',  # Same color as the scatter plot
        text=text_labels,
        textposition='auto',
        customdata=[f'{value} tasks ({value/total_tasks:.1%} of failures)' for value in values],
        textfont=dict(color='black', size=14, family='Arial', weight=2),
        hovertemplate='<b>%{y}</b><br>' +
                      'Affected Tasks: %{customdata}<extra></extra>'
    )])

    # Calculate dynamic height based on number of categories
    height = max(400, len(categories) * 150)  # At least 400px, or 150px per category

    fig.update_layout(
        height=height,  # Dynamic height
        margin=dict(l=20, r=20, t=20, b=20),  # Reduced margins
        xaxis=dict(
            showline=True,
            linecolor='black',
            showgrid=False,
            side='top'  # Move x-axis to top
        ),
        yaxis=dict(
            showline=True,
            linecolor='black',
            showgrid=False,
            autorange="reversed"  # This will put the category with the highest value at the top
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        bargap=0.2,
        bargroupgap=0.1,
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        modebar=dict(
            activecolor='#1f77b4',
            orientation='h',
            bgcolor='rgba(255,255,255,0.8)',
            color='#777',
            add=['pan2d'],
            remove=[
                'zoom2d', 'zoomIn2d', 'zoomOut2d', 'resetScale2d',
                'hoverClosestCartesian', 'hoverCompareCartesian',
                'toggleSpikelines', 'lasso2d', 'lasso', 'select2d', 'select'
            ]
        ),
        dragmode='pan'
    )

    return fig

def create_scatter_plot(df, x: str, y: str, x_label: str = None, y_label: str = None, hover_data: list = None, cost_cutoff_multiplier: float = None, return_cutoff_applied: bool = False):
    # Extract model from agent name if token usage is not available
    def extract_model(agent_name):
        # Remove URL part if present
        if '[' in agent_name:
            agent_name = agent_name.split(']')[0].split('[')[1]
        if '(' in agent_name and ')' in agent_name:
            model = agent_name.split('(')[-1].rstrip(')')
            if any(model_prefix in model.lower() for model_prefix in ['gpt-', 'claude-', 'gemini-', 'meta-llama', 'openai', 'anthropic', 'o1-', 'o3-']):
                return model.strip()
        return 'Other'

    # Map model names to developers
    def get_model_developer(model_name):
        model_lower = model_name.lower()
        if any(prefix in model_lower for prefix in ['gpt', 'o1', 'o3', 'o4', 'openai']):
            return 'OpenAI'
        elif any(prefix in model_lower for prefix in ['claude', 'anthropic']):
            return 'Anthropic'
        elif any(prefix in model_lower for prefix in ['gemini', 'google']):
            return 'Google'
        elif any(prefix in model_lower for prefix in ['deepseek']):
            return 'DeepSeek'
        else:
            return 'Other'

    # Compute per-agent means and Pareto frontier
    unique_agents = df['Agent Name'].unique()
    agent_means = {
        agent: (
            df[df['Agent Name'] == agent]['Total Cost'].mean(),
            df[df['Agent Name'] == agent]['Accuracy'].mean()
        )
        for agent in unique_agents
    }
    agents = [Agent(cost, acc) for cost, acc in agent_means.values()]
    pareto_frontier = compute_pareto_frontier(agents)

    # Set of rounded frontier points for robust matching
    frontier_points = {(round(a.total_cost, 6), round(a.accuracy, 6)) for a in pareto_frontier}
    
    # Calculate cost cutoff if specified
    max_x_range = None
    cutoff_applied = False
    if cost_cutoff_multiplier is not None:
        # Find the maximum accuracy
        max_accuracy = max(acc for _, acc in agent_means.values())
        
        # Define accuracy tolerance (e.g., within 0.5 percentage points)
        accuracy_tolerance = 0.5
        
        # Find all agents within tolerance of max accuracy
        top_performers = [
            (cost, acc) for cost, acc in agent_means.values() 
            if abs(acc - max_accuracy) <= accuracy_tolerance
        ]
        
        if top_performers:
            # Use the cheapest among top performers as reference
            reference_cost = min(cost for cost, _ in top_performers)
            cutoff_threshold = reference_cost * cost_cutoff_multiplier
            
            # Check if any agent has cost beyond the threshold
            max_cost = max(cost for cost, _ in agent_means.values())
            if max_cost > cutoff_threshold:
                # Only apply cutoff if there are actually points beyond it
                max_x_range = cutoff_threshold
                cutoff_applied = True

    fig = go.Figure()

    # Pareto frontier line
    pareto_points = sorted([(a.total_cost, a.accuracy) for a in pareto_frontier], key=lambda p: p[0])
    fig.add_trace(go.Scatter(
        x=[p[0] for p in pareto_points],
        y=[p[1] for p in pareto_points],
        mode='lines',
        name='Pareto Frontier',
        hoverinfo=None,
        line=dict(color='black', width=1, dash='dash')
    ))

    # Colors by developer instead of individual models
    color_sequence = px.colors.qualitative.Dark2
    unique_developers = ['OpenAI', 'Anthropic', 'Google', 'DeepSeek', 'Other']
    developer_color_map = {dev: color_sequence[i % len(color_sequence)] for i, dev in enumerate(unique_developers)}
    
    # Track which developers we've seen to manage legend
    developers_seen = set()

    # Plot each agent (markers + optional annotation if on frontier)
    unique_agents = df[hover_data[0]].unique()

    for agent in unique_agents:
        agent_data = df[df[hover_data[0]] == agent]

        # Clean tooltip name (remove URL if present)
        if '[' in str(agent_data['Agent Name'].iloc[0]):
            agent_data.loc[:, 'Agent Name'] = agent_data['Agent Name'].str.rsplit(']').str[0].str[1:]

        x_value = [float(np.mean(agent_data[x].values))]
        y_value = [float(np.mean(agent_data[y].values))]
        model = agent_data['Model Name'].iloc[0]
        developer = get_model_developer(model)

        # Error bars (if multiple runs)
        if len(agent_data) > 1:
            fig.add_trace(go.Scatter(
                x=x_value, y=y_value,
                error_x=dict(type='data', symmetric=False,
                             array=[np.max(agent_data[x]) - x_value[0]],
                             arrayminus=[x_value[0] - np.min(agent_data[x])],
                             color='#fec44f', thickness=1.5, width=4),
                mode='markers', marker=dict(size=1, color='#fec44f'),
                showlegend=False, hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=x_value, y=y_value,
                error_y=dict(type='data', symmetric=False,
                             array=[np.max(agent_data[y]) - y_value[0]],
                             arrayminus=[y_value[0] - np.min(agent_data[y])],
                             color='#bdbdbd', thickness=1.5, width=4),
                mode='markers', marker=dict(size=1, color='#bdbdbd'),
                showlegend=False, hoverinfo='skip'
            ))

        # Marker for this agent
        # Only show in legend if this is the first time we see this developer
        show_in_legend = developer not in developers_seen and developer in unique_developers
        if show_in_legend:
            developers_seen.add(developer)
        
        fig.add_trace(go.Scatter(
            x=x_value,
            y=y_value,
            mode='markers',
            name=developer,  # Show developer name in legend instead of model
            marker=dict(size=10, color=developer_color_map.get(developer, '#1f77b4')),
            customdata=agent_data[hover_data],
            showlegend=show_in_legend,  # Only show first occurrence of each developer
            legendgroup=developer,  # Group by developer
            hovertemplate="<br>".join([
                "<b>Agent</b>: %{customdata[0]}",
                "<b>Model</b>: " + model,  # Keep detailed model info in hover
                "<b>Total Cost</b>: $%{x:.1f}",
                "<b>Accuracy</b>: %{y:.1f}%<extra></extra>",
            ]),
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        ))

        # Annotate ONLY if this point is on the Pareto frontier
        if (round(x_value[0], 6), round(y_value[0], 6)) in frontier_points:
            label = agent_data['Agent Name'].iloc[0]
            fig.add_annotation(
                x=x_value[0],
                y=y_value[0],
                text=label,
                showarrow=True,
                arrowhead=0,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="#CCCCCC",
                ax=60,   # modest offset
                ay=20,
                font=dict(size=10),
            )

    # Legend entries for CI
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(color='#fec44f', size=10),
        name='Cost CI (Min-Max)'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(color='#bdbdbd', size=10),
        name='Accuracy CI (Min-Max)'
    ))

    fig.update_layout(
        height=600,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(showline=True, linecolor='black', showgrid=False),
        yaxis=dict(showline=True, showgrid=False, linecolor='black'),
        plot_bgcolor='white',
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.98, bgcolor="rgba(255, 255, 255, 0.5)"),
        modebar=dict(
            activecolor='#1f77b4',
            orientation='h',
            bgcolor='rgba(255,255,255,0.8)',
            color='#777',
            add=['pan2d'],
            remove=['zoom2d','zoomIn2d','zoomOut2d','resetScale2d','hoverClosestCartesian','hoverCompareCartesian','toggleSpikelines','lasso2d','lasso','select2d','select']
        ),
        dragmode='pan',
        showlegend=True,
        annotations=list(fig.layout.annotations)  # keep added annotations
    )

    fig.update_yaxes(rangemode="tozero")
    if max_x_range is not None:
        fig.update_xaxes(rangemode="tozero", range=[0, max_x_range])
    else:
        fig.update_xaxes(rangemode="tozero")
    
    if return_cutoff_applied:
        return fig, cutoff_applied
    return fig


import plotly.graph_objects as go
import textwrap

def create_flow_chart(steps):
    node_x = []
    node_y = []
    edge_x = []
    edge_y = []
    node_text = []
    hover_text = []
    node_colors = []
    node_shapes = []
    
    # Define color and shape mappings
    color_map = {True: 'green', False: 'red'}  # True for success, False for challenges
    shape_map = {
        'plan': 'octagon',
        'tool': 'square',
        'retrieve': 'diamond',
        'other': 'circle'
    }
    
    for i, step in enumerate(steps):
        node_x.append(i)
        node_y.append(0)
        
        # Extract Description, Assessment, and new attributes
        analysis = step['analysis']
        if isinstance(analysis, str):
            try:
                analysis = json.loads(analysis)
            except json.JSONDecodeError:
                analysis = {}
        
        description = analysis.get('description', 'No description available.')
        assessment = analysis.get('assessment', 'No assessment available.')
        success = analysis.get('success', True)  # Assuming True if not specified
        # action_type = analysis.get('action_type', 'other')  # Default to 'other' if not specified
        step_headline = analysis.get('headline', '')
        
        # Set node color and shape based on attributes
        node_colors.append(color_map[success])
        # node_shapes.append(shape_map.get(action_type, 'circle'))
        
        # Wrap text to improve readability
        wrapped_description = '<br>'.join(textwrap.wrap(description, width=90, max_lines=20))
        wrapped_assessment = '<br>'.join(textwrap.wrap(assessment, width=90, max_lines=10))
        wrapped_outline = textwrap.shorten(step_headline, width=50, placeholder='')
        wrapped_outline = '' if wrapped_outline == '' else f": {wrapped_outline}"

        node_text_outline = '' if wrapped_outline == '' else f":<br>{'<br>'.join(textwrap.wrap(step_headline, width=30, placeholder=''))}"
        node_text.append(f"Step {i+1}{node_text_outline}")
        
        # Create formatted hover text without indentation
        hover_info = f"<b>Step {i+1}{wrapped_outline}</b><br><br>" \
                     f"<b>Description:</b><br>" \
                     f"{wrapped_description}<br><br>" \
                    #  f"<b>Assessment:</b><br>" \
                    #  f"{wrapped_assessment}<br><br>" \
                    #  f"<b>Successful:</b> {'Yes' if success else 'No'}<br>" \
                    #  f"<b>Action Type:</b> {action_type.capitalize()}"
        hover_text.append(hover_info)
        
        if i > 0:
            edge_x.extend([i-1, i, None])
            edge_y.extend([0, 0, None])
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        showlegend=False,
        hovertext=hover_text,
        hoverinfo='text',
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        marker=dict(
            # color=node_colors,
            color='#3498db',
            size=30,
            line_width=2,
            # symbol=node_shapes
        ))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        showlegend=False,
        mode='lines')
    
    # Create legend traces
    legend_traces = []
    
    # # Color legend
    # for success, color in color_map.items():
    #     legend_traces.append(go.Scatter(
    #         x=[None], y=[None],
    #         mode='markers',
    #         marker=dict(size=10, color=color),
    #         showlegend=True,
    #         name=f"{'Success' if success else 'Issue'}"
    #     ))
    
    # # Shape legend
    # for action, shape in shape_map.items():
    #     legend_traces.append(go.Scatter(
    #         x=[None], y=[None],
    #         mode='markers',
    #         marker=dict(size=10, symbol=shape, color='gray'),
    #         showlegend=True,
    #         name=f"{action.capitalize()}"
    #     ))

    # Combine all traces
    all_traces = [edge_trace, node_trace] + legend_traces

    layout = go.Layout(
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        modebar=dict(
            activecolor='#1f77b4',  # Color of active tool
            orientation='h',  # Vertical orientation
            bgcolor='rgba(255,255,255,0.8)',  # Slightly transparent white background
            color='#777',  # Color of inactive tools
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
    )
    
    fig = go.Figure(data=all_traces, layout=layout)
    
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor='rgba(255,255,255,0.8)',  # Set legend background to slightly transparent white
        bordercolor='rgba(0,0,0,0.1)',  # Add a light border to the legend
        borderwidth=1
    ),
    dragmode='pan'
    )

    config = {
        'add': ['pan2d'],
        'remove': [
            'zoom2d', 
            'zoomIn2d', 
            'zoomOut2d', 
            'resetScale2d',
            'hoverClosestCartesian', 
            'hoverCompareCartesian',
            'toggleSpikelines',
            'lasso2d',
            'lasso',
            'select2d',
            'select',
        ]
    }
    
    # Apply the config to the figure
    fig.update_layout(modebar=config)
    
    return fig

def create_model_timeline_chart(leaderboard_df, benchmark_name):
    """
    Create a timeline chart showing model accuracies over time of release.
    """
    import re
    
    def extract_release_date(model_name):
        """Extract release date from model name in format '(Month YYYY)' or 'Month YYYY'"""
        # Pattern to match (Month YYYY) or just Month YYYY at the end
        pattern = r'(?:\(([A-Za-z]+ \d{4})\))|([A-Za-z]+ \d{4})$'
        match = re.search(pattern, model_name)
        
        if match:
            # Get the matched group (either from parentheses or end of string)
            date_str = match.group(1) if match.group(1) else match.group(2)
            return date_str
        return None
    
    def convert_date_to_format(date_str):
        """Convert 'Month YYYY' format to 'YYYY-MM' format"""
        if not date_str:
            return None
            
        # Month name to number mapping
        month_map = {
            'January': '01', 'February': '02', 'March': '03', 'April': '04',
            'May': '05', 'June': '06', 'July': '07', 'August': '08',
            'September': '09', 'October': '10', 'November': '11', 'December': '12'
        }
        
        try:
            parts = date_str.split()
            if len(parts) == 2:
                month_name, year = parts
                month_num = month_map.get(month_name)
                if month_num:
                    return f"{year}-{month_num}"
        except:
            pass
        return None
    
    # Extract model release dates from DEFAULT_PRICING
    model_release_dates = {}
    for model_name in DEFAULT_PRICING.keys():
        release_date_str = extract_release_date(model_name)
        formatted_date = convert_date_to_format(release_date_str)
        if formatted_date:
            model_release_dates[model_name] = formatted_date
    
    # Color mapping for different providers (same as pareto chart)
    color_sequence = px.colors.qualitative.Dark2
    unique_developers = ['OpenAI', 'Anthropic', 'Google', 'DeepSeek', 'Other']
    provider_colors = {dev: color_sequence[i % len(color_sequence)] for i, dev in enumerate(unique_developers)}
    
    def get_provider(model_name):
        """Determine provider from model name (same logic as pareto chart)"""
        model_lower = model_name.lower()
        if any(prefix in model_lower for prefix in ['gpt', 'o1', 'o3', 'o4', 'openai']):
            return 'OpenAI'
        elif any(prefix in model_lower for prefix in ['claude', 'anthropic', 'sonnet']):
            return 'Anthropic'
        elif any(prefix in model_lower for prefix in ['gemini', 'google']):
            return 'Google'
        elif any(prefix in model_lower for prefix in ['deepseek']):
            return 'DeepSeek'
        else:
            return 'Other'
    
    # Prepare data for the chart
    timeline_data = []
    
    # Get the best accuracy for each model in the leaderboard
    model_accuracies = leaderboard_df.groupby('Models')['Accuracy'].max().reset_index()
    
    for _, row in model_accuracies.iterrows():
        model_name = row['Models']
        accuracy = row['Accuracy']
        
        # Use exact matching only
        if model_name in model_release_dates:
            release_date = model_release_dates[model_name]
            provider = get_provider(model_name)
            timeline_data.append({
                'model': model_name,
                'accuracy': accuracy,
                'release_date': release_date,
                'provider': provider,
                'color': provider_colors[provider]
            })
    
    if not timeline_data:
        return None
    
    # Convert to DataFrame and sort by date
    timeline_df = pd.DataFrame(timeline_data)
    timeline_df['date_sort'] = pd.to_datetime(timeline_df['release_date'])
    timeline_df = timeline_df.sort_values('date_sort')
    
    # Create the timeline chart
    fig = go.Figure()
    
    # Sort timeline data by accuracy for collision detection
    timeline_df_sorted = timeline_df.sort_values(['accuracy', 'release_date'])
    
    # Calculate label positions to avoid collisions
    label_positions = []
    position_cycle = ['middle left', 'middle right', 'top center', 'bottom center']
    accuracy_threshold = 2.0  # Consider models within 2% accuracy as potentially colliding
    
    for i, row in timeline_df_sorted.iterrows():
        current_accuracy = row['accuracy']
        
        # Special case: if accuracy is 0%, always put label on top
        if current_accuracy == 0:
            position = 'top center'
        else:
            # Find how many models have similar accuracy (within threshold)
            similar_models = timeline_df_sorted[
                (abs(timeline_df_sorted['accuracy'] - current_accuracy) <= accuracy_threshold)
            ]
            
            if len(similar_models) == 1:
                # Only one model at this accuracy level, use default position
                position = 'middle left'
            else:
                # Multiple models with similar accuracy, cycle through positions
                model_index = list(similar_models.index).index(i)
                position = position_cycle[model_index % len(position_cycle)]
        
        label_positions.append(position)
    
    # Create a mapping from original index to position
    position_map = dict(zip(timeline_df_sorted.index, label_positions))
    
    # Add traces for each provider
    for provider in timeline_df['provider'].unique():
        provider_data = timeline_df[timeline_df['provider'] == provider]
        
        # Clean model names and get positions for this provider
        clean_model_names = []
        text_positions = []
        for idx, model in zip(provider_data.index, provider_data['model']):
            clean_name = model.split('(')[0].strip()
            clean_model_names.append(clean_name)
            text_positions.append(position_map[idx])
        
        fig.add_trace(go.Scatter(
            x=provider_data['release_date'],
            y=provider_data['accuracy'],
            mode='markers+text',
            name=provider,
            marker=dict(
                size=12,
                color=provider_colors[provider],
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=clean_model_names,
            textposition=text_positions,
            textfont=dict(size=10, color='black'),
            hovertemplate='<b>%{text}</b><br>Release: %{x}<br>Accuracy: %{y:.1f}%<extra></extra>'
        ))
    
    # Calculate y-axis max value
    max_accuracy = timeline_df['accuracy'].max()
    y_max = int((max_accuracy + 10) // 10 * 10 + 10)  # Round up to nearest 10 after adding 10
    
    # Calculate x-axis max value (max release date + 2 months)
    max_date = pd.to_datetime(timeline_df['release_date'].max())
    x_max_date = max_date + pd.DateOffset(months=2)
    x_max = x_max_date.strftime('%Y-%m')
    
    # Update layout
    fig.update_layout(
        xaxis_title='Release Date',
        yaxis_title='Accuracy (%)',
        height=500,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=40, b=60, l=60, r=20),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Customize x-axis to show dates nicely and start from November 2024
    fig.update_xaxes(
        tickformat='%b %Y',
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.3)',
        dtick='M1',  # Monthly ticks
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='lightgray',
        range=['2024-11', x_max]  # Start from November 2024, end 2 months after max date
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.3)',
        dtick=5,  # Step of 5 for y-axis
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='lightgray',
        range=[0, y_max]  # Set custom range from 0 to calculated max
    )
    
    return fig