import plotly.graph_objects as go
import plotly.colors as pc
import numpy as np
import pandas as pd
import src.load as load
from plotly.subplots import make_subplots

def get_colors_sequence(name,n):
    for module in (pc.sequential, pc.diverging, pc.cyclical):
        if hasattr(module, name):
            color_scale = getattr(module, name)
            indices = np.linspace(0, len(color_scale)-1, n, dtype=int)
            colors = [color_scale[i] for i in indices]
            return colors
    raise ValueError(f"Colorscale '{name}' not found in Plotly colors.")

def this_season_graph(szn_act_kg,szn_adj_kg):
    total_actuals = szn_act_kg.sum(axis=0)
    total_preds = szn_adj_kg.sum(axis=0)

    x_vals = np.arange(53)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_vals, y=total_preds[0], mode='lines', name='Predicted', line={'color': 'red', 'dash': 'dash'}))
    fig.add_trace(go.Bar(x=x_vals[1:], y=total_actuals[1:], marker_color='black',opacity=0.5,showlegend=False))
    fig.add_trace(go.Bar(x=x_vals[:1], y=total_actuals[:1], marker_color='black',name='Actuals'))

    fig.add_vline(x=0, line_width=2, line_dash="dot", line_color="black", annotation_text="Current", annotation_position="top right")

    steps = []
    for i in range(51):
        step = dict(
            method="update",
            args=[
                {
                    "x": [x_vals,x_vals[i:],x_vals[:i]],
                    "y": [total_preds[i],total_actuals[i:],total_actuals[:i]]
                },
                {"shapes[0].x0": i, "shapes[0].x1": i}  # move vline
                ],
            label=f"Day {i+1}")
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Day: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(sliders=sliders)
    return fig

def plot_season(actuals,predictions,classes,color_name):
    c_scale = get_colors_sequence(color_name,len(classes))
    total_actuals  = actuals.sum(axis=0)
    total_predictions = predictions.sum(axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = np.arange(40), y=total_actuals, mode='lines', name='Total Actual', line={'color':'black'}))
    fig.add_trace(go.Scatter(x = np.arange(40), y=total_predictions, mode='lines', name='Total Pred', line={'color':'red', 'dash':'dash'}))
    for i,c in enumerate(classes):
        fig.add_trace(go.Bar(x = np.arange(40), y=actuals[i], name=f"{c} Actual", marker_color=c_scale[i]))
        fig.add_trace(go.Scatter(x = np.arange(40), y=predictions[i], mode='lines', name=f"{c} Pred", line={'color':c_scale[i], 'dash':'dash'}))
    fig.update_layout(
        title='Actual vs Predicted Kilos'
    )
    return fig

def plot_diffs(diff,classes,color_name):
    agg_diff = diff.cumsum(axis=1)
    c_scale = get_colors_sequence(color_name,len(classes))

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Difference by Class', 'Cumulative Difference by Class'))

    # Plot diff in first subplot
    for i, c in enumerate(classes):
        fig.add_trace(
            go.Bar(x=np.arange(40), y=diff[i], name=f"{c} Difference", marker_color=c_scale[i], showlegend=(i==0)),
            row=1, col=1
        )   

    # Plot agg_diff in second subplot
    for i, c in enumerate(classes):
        fig.add_trace(
            go.Scatter(x=np.arange(40), y=agg_diff[i], mode='lines', name=f"{c} Cumulative Diff", line={'color':c_scale[i], 'dash':'dash'}, showlegend=False),
            row=1, col=2
        )

    fig.update_layout(
        title_text='Difference and Cumulative Difference between Actual and Predicted Kilos'
    )
    return fig

def pred_evolution(this_week,szn_act_kg,szn_init_kg,szn_adj_kg,idx_dict,color_seq = 'Sunset'):
    f"""
    Creates the figure of the graph of the evolution of the preds for week [this_week]. Shows initial preds, and actuals on the graph, broken down by class.

    args:
    this_week [int]: Week of the season
    szn_act_kg [array(N,OutputDim)]: actual kilograms for the season for each batch
    szn_init_kg [array(N,OutputDim)]: initial predicted kilograms for each batch
    szn_adj_kg [array(N,Num_Weeks,OutputDim)]: adjusted predictions for each batch for each week in Num_Weeks
    idx_dict [col: mappings]: Mappings for class, transplant_week etc. boolean masks.
    color_seq [str]: name of the colorscheme

    returns:
    fig
    """
    adj_preds = (load.collapse_table(szn_adj_kg,idx_dict,'class'))[:,:this_week,this_week]
    init_preds = load.collapse_table(szn_init_kg[:,this_week],idx_dict,'class')
    act = load.collapse_table(szn_act_kg[:,this_week],idx_dict,'class')
    
    classes = list(idx_dict['class'].keys())
    colors = get_colors_sequence(color_seq,len(classes))

    fig = go.Figure()
    x_vals = np.arange(this_week)
    start_end = [0,this_week]

    for i,(c,color) in enumerate(zip(classes,colors)):
        #add the adjusted_preds
        fig.add_trace(go.Scatter(x = x_vals, y=adj_preds[i], mode='lines',name=c,line={'color':color}))

        #add the init_preds
        fig.add_trace(go.Scatter(
            x=start_end,
            y=[init_preds[i], init_preds[i]],
            mode='lines',
            line={"color": color, "width": 2, "dash": "dash"},
            name=f'initial preds for {c}',
            showlegend=False,
        ))

        #add the actuals
        fig.add_trace(go.Scatter(
            x=start_end,
            y=[act[i], act[i]],
            mode='lines',
            line={"color": color, "width": 2, "dash": "dot"},
            name=f'actual kg for {c}',
            showlegend=False,
        ))
    fig.update_layout(
        title=f"Evolution of Predictions for Week {this_week}",
        legend={
            'orientation':'h',
            'yanchor':'top',
            'y':-0.15,
            'xanchor':'center',
            'x':0.5}
        )
    return fig
