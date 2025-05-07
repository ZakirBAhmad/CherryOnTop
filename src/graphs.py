import plotly.graph_objects as go
from plotly.colors import sample_colorscale

def graph_preds_reg(df, af, title, color_palette='Viridis'):
    """
    desc:
        Create a plotly figure with predicted and actual values for regression data.
    
    params:
        df: DataFrame containing predicted values with MultiIndex columns
        af: DataFrame containing actual values with MultiIndex columns
        title: Title for the graph
        color_palette: Name of plotly colorscale. Defaults to 'Viridis'.
    
    returns:
        Interactive figure with dropdown menu to filter traces
    """
    fig = go.Figure()
    x_values = df.index
    colors = sample_colorscale(color_palette, [i / (len(df.columns) - 1) for i in range(len(df.columns))])
    color_map = {key: colors[i] for i, key in enumerate(df.columns)}

    for i, col in enumerate(df.columns):
        label = col[0] + ': ' + col[1]
        fig.add_trace(go.Scatter(x=x_values, y=df[col], name=label + ' Predicted', line=dict(color=color_map[col])))
        fig.add_trace(go.Bar(x=x_values, y=af[col], name= label + ' Actual', marker_color=color_map[col], opacity=0.75))

    buttons = []
    for i, key in enumerate(df.columns):
        # Each key creates 2 traces (scatter + bar), so need to toggle both
        visibility = [j//2 == i for j in range(len(df.columns)*2)]  # Show only the i-th pair of traces
        buttons.append(dict(
            label=f"{key[0]}: {key[1]}",
            method='update',
            args=[{'visible': visibility}, {'title': f'Selected: {key[0]}: {key[1]}'}]
        ))

    # Add "Show All" button
    buttons.append(dict(
        label='Show All', 
        method='update',
        args=[{'visible': [True]*(len(df.columns)*2)}, {'title': 'All Traces Visible'}]
    ))

    # Add dropdown menu to layout
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction='down',
            showactive=True,
            x=1.05,
            xanchor='left',
            y=1,
            yanchor='top'
        )],
        title=title,
        template='plotly_white',
        xaxis_title='Week after Transplant',
        yaxis_title='Harvest (kg)'
    )