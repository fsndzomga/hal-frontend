import json
import plotly.express as px
from utils.pareto import Agent, compute_pareto_frontier
import plotly.graph_objects as go
import textwrap
import numpy as np
import pandas as pd
from scipy import stats
from plotly.subplots import make_subplots
from utils.db import DEFAULT_PRICING


def create_leaderboard(df, benchmark_name = None):
    df = df.groupby('Agent Name', as_index=False).agg({
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
                'Non-Refusal Harm Score': 'mean'  # Preserve URL
            })
    
    # Sort by Accuracy in descending order
    if benchmark_name == 'agentharm':
        df = df.sort_values('Accuracy', ascending=True)
    else:
        df = df.sort_values('Accuracy', ascending=False)
    
    # cast dtypes to string
    df = df.astype(str)

    # Count number of runs per agent
    runs_per_agent = df.groupby('Agent Name').size().reset_index()
    runs_per_agent.columns = ['Agent Name', 'Runs']
    runs_per_agent.drop("Runs",inplace=True, axis=1)
    
    # Merge runs count back into the dataframe
    df = df.merge(runs_per_agent, on='Agent Name', how='left')
    
    # Add model names to leaderboard data
    df['Models'] = df['Agent Name'].apply(lambda x: x.split('(')[-1].rstrip(')') if any(model in x for model in DEFAULT_PRICING.keys()) else "")
    # Remove model names from agent name
    df['Agent Name'] = df['Agent Name'].apply(lambda x: '('.join(x.split('(')[:-1]).strip() if '(' in x else x)
        

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

def create_scatter_plot(df, x: str, y: str, x_label: str = None, y_label: str = None, hover_data: list = None):
    # Extract model from agent name if token usage is not available
    def extract_model(agent_name):
        # Remove URL part if present
        if '[' in agent_name:
            agent_name = agent_name.split(']')[0].split('[')[1]
        
        # Look for model name in parentheses at the end
        if '(' in agent_name and ')' in agent_name:
            model = agent_name.split('(')[-1].rstrip(')')
            if any(model_prefix in model.lower() for model_prefix in ['gpt-', 'claude-', 'gemini-', 'meta-llama', 'openai', 'anthropic', 'o1-']):
                return model.strip()
        return 'Other'

    df['Model'] = df['Agent Name'].apply(extract_model)

    # Create agents using the values directly from the DataFrame
    # The DataFrame should already have aggregated values
    unique_agents = df['Agent Name'].unique()
    agents = [Agent(df[df['Agent Name'] == agent]['Total Cost'].mean(), df[df['Agent Name'] == agent]['Accuracy'].mean()) for agent in unique_agents]
    pareto_frontier = compute_pareto_frontier(agents)

    fig = go.Figure()

    # Sort the Pareto frontier points by x-coordinate
    pareto_points = sorted([(agent.total_cost, agent.accuracy) for agent in pareto_frontier], key=lambda x: x[0])
    # Add the Pareto frontier line
    fig.add_trace(go.Scatter(
        x=[point[0] for point in pareto_points],
        y=[point[1] for point in pareto_points],
        mode='lines',
        name='Pareto Frontier',
        hoverinfo=None,
        line=dict(color='black', width=1, dash='dash')
    ))

    # Define color map for models
    color_sequence = px.colors.qualitative.Dark2
    unique_models = sorted(df['Model'].unique())
    color_map = {model: color_sequence[i % len(color_sequence)] for i, model in enumerate(unique_models)}

    # Plot scatter points and error bars for each agent
    unique_agents = df[hover_data[0]].unique()
    
    # Create lists to store all point coordinates for label placement
    all_x = []
    all_y = []
    all_labels = []
    
    # Keep track of which models we've already added to the legend
    models_in_legend = set()
    
    for agent in unique_agents:
        agent_data = df[df[hover_data[0]] == agent]
                
        # remove url from tooltip name
        if '[' in str(agent_data['Agent Name'].iloc[0]):
            agent_data.loc[:, 'Agent Name'] = agent_data['Agent Name'].str.rsplit(']').str[0].str[1:]

        x_value = [np.mean(agent_data[x].values)]
        y_value = [np.mean(agent_data[y].values)]
        model = agent_data['Model'].iloc[0]
        
        # Store coordinates and label for later use
        all_x.extend(x_value)
        all_y.extend(y_value)
        all_labels.extend([agent_data['Agent Name'].iloc[0]])

        if len(agent_data) > 1:
            # Add error bars for x (cost minmax)
            fig.add_trace(go.Scatter(
                x=x_value,
                y=y_value,
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=[np.max(agent_data[x]) - x_value[0]],
                    arrayminus=[x_value[0] - np.min(agent_data[x])],
                    color='#fec44f',
                    thickness=1.5,
                    width=4,
                ),
                mode='markers',
                marker=dict(size=1, color='#fec44f'),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Add error bars for y (accuracy minmax)
            fig.add_trace(go.Scatter(
                x=x_value,
                y=y_value,
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[np.max(agent_data[y]) - y_value[0]],
                    arrayminus=[y_value[0] - np.min(agent_data[y])],
                    color='#bdbdbd',
                    thickness=1.5,
                    width=4,
                ),
                mode='markers',
                marker=dict(size=1, color='#bdbdbd'),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add scatter points for this agent
        show_in_legend = model not in models_in_legend
        if show_in_legend:
            models_in_legend.add(model)
            
        fig.add_trace(go.Scatter(
            x=x_value,
            y=y_value,
            mode='markers',
            name=model,  # Use model as the name for legend grouping
            marker=dict(size=10, color=color_map[model]),
            customdata=agent_data[hover_data],
            showlegend=show_in_legend,
            hovertemplate="<br>".join([
                "<b>Agent</b>: %{customdata[0]}",
                "<b>Model</b>: " + model,
                "<b>Total Cost</b>: $%{x:.1f}",
                "<b>Accuracy</b>: %{y:.1f}%<extra></extra>",
            ]),
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        ))

    # Add legend entries for error bars
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

    # Update layout to handle overlapping labels
    fig.update_layout(
        height=600,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(
            showline=True,
            linecolor='black',
            showgrid=False
        ),
        yaxis=dict(
            showline=True,
            showgrid=False,
            linecolor='black'
        ),
        plot_bgcolor='white',
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255, 255, 255, 0.5)"
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
        # Add configuration for handling overlapping labels
        showlegend=True,
        annotations=[],
    )

    # Add non-overlapping labels using annotations
    for i in range(len(all_x)):
        # Default position: lower right
        ax = 20
        ay = 20
        
        # Adjust position if near axes
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)
        
        # If point is near minimum x-axis (left side)
        if all_x[i] < min(all_x) + 0.05 * x_range:
            ax = 120  # Large shift for points very close to left axis
            
        # If point is near maximum x-axis (right side)
        if all_x[i] > max(all_x) - 0.1 * x_range:
            ax = -20  # Move label to the left
        
        # If point is near minimum y-axis (bottom)
        if all_y[i] < min(all_y) + 0.05 * y_range:
            ay = -30  # Move label up
        
        # If point is near maximum y-axis (top)
        if all_y[i] > max(all_y) - 0.1 * y_range:
            ay = -20  # Move label down
               
        # Check for overlap with previous labels
        overlap = False
        for j in range(i):
            # Simple distance check between points
            dx = abs(all_x[i] - all_x[j])
            dy = abs(all_y[i] - all_y[j])
            
            # Reduced overlap threshold from 0.2 to 0.1
            if dx < 0.12 * x_range and dy < 0.12 * y_range:
                # If points are close, try different positions
                if not overlap:
                    ax += 20  # Smaller increment (from 40 to 20)
                    ay += 20  # Smaller increment (from 40 to 20)
                overlap = True
        
        if not overlap or i == 0:  # Always show first label
            fig.add_annotation(
                x=all_x[i],
                y=all_y[i],
                text=all_labels[i],
                showarrow=True,
                arrowhead=0,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="#CCCCCC",
                ax=ax,
                ay=ay,
                font=dict(
                    size=10
                ),
            )

    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(rangemode="tozero")

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