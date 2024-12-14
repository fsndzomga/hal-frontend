import json
from pathlib import Path
import pandas as pd
import plotly.express as px
from utils.pareto import Agent, compute_pareto_frontier
import plotly.graph_objects as go
import textwrap

def create_scatter_plot(df, x: str, y: str, x_label: str = None, y_label: str = None, hover_data: list = None):
    agents = [Agent(row.results_total_cost, row.results_accuracy) for row in df.itertuples()]
    pareto_frontier = compute_pareto_frontier(agents)


    fig = px.scatter(df, 
                     x=x, 
                     y=y,
                     hover_data=hover_data,
                     )
    

    # Sort the Pareto frontier points by x-coordinate
    pareto_points = sorted([(agent.total_cost, agent.accuracy) for agent in pareto_frontier], key=lambda x: x[0])
    
    # Add the Pareto frontier line
    fig.add_trace(go.Scatter(
        x=[point[0] for point in pareto_points],
        y=[point[1] for point in pareto_points],
        mode='lines',
        name='Pareto Frontier',
        line=dict(color='black', width=1, dash='dash')
    ))

    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(rangemode="tozero")

    fig.update_layout(
    width = 600,
    height = 500,
    xaxis_title = x_label,
    yaxis_title = y_label,
    xaxis = dict(
        showline = True,
        linecolor = 'black',
        showgrid = False),
    yaxis = dict(
        showline = True,
        showgrid = False,
        linecolor = 'black'),
    plot_bgcolor = 'white',
    # Legend positioning
    legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.98,
        bgcolor="rgba(255, 255, 255, 0.5)"  # semi-transparent white background
        )
    )
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
        action_type = analysis.get('action_type', 'other')  # Default to 'other' if not specified
        step_outline = analysis.get('step_outline', '')
        
        # Set node color and shape based on attributes
        node_colors.append(color_map[success])
        node_shapes.append(shape_map.get(action_type, 'circle'))
        
        # Wrap text to improve readability
        wrapped_description = '<br>'.join(textwrap.wrap(description, width=50))
        wrapped_assessment = '<br>'.join(textwrap.wrap(assessment, width=50))
        wrapped_outline = textwrap.shorten(step_outline, width=30, placeholder='')
        wrapped_outline = '' if wrapped_outline == '' else f": {wrapped_outline}"

        node_text_outline = '' if wrapped_outline == '' else f":<br>{textwrap.shorten(step_outline, width=30, placeholder='')}"
        node_text.append(f"Step {i+1}{node_text_outline}")
        
        # Create formatted hover text without indentation
        hover_info = f"<b>Step {i+1}{wrapped_outline}</b><br><br>" \
                     f"<b>Description:</b><br>" \
                     f"{wrapped_description}<br><br>" \
                     f"<b>Assessment:</b><br>" \
                     f"{wrapped_assessment}<br><br>" \
                     f"<b>Successful:</b> {'Yes' if success else 'No'}<br>" \
                     f"<b>Action Type:</b> {action_type.capitalize()}"
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
            color=node_colors,
            size=30,
            line_width=2,
            symbol=node_shapes
        ))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        showlegend=False,
        mode='lines')
    
    # Create legend traces
    legend_traces = []
    
    # Color legend
    for success, color in color_map.items():
        legend_traces.append(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            showlegend=True,
            name=f"{'Success' if success else 'Issue'}"
        ))
    
    # Shape legend
    for action, shape in shape_map.items():
        legend_traces.append(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, symbol=shape, color='gray'),
            showlegend=True,
            name=f"{action.capitalize()}"
        ))

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