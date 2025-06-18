import numpy as np
import plotly.graph_objects as go


def graph_harvest(preds,actuals):
    summed_preds = np.sum(preds, axis=0)  # Shape: (20, 51)
    summed_actuals = np.sum(actuals, axis=0) 

    # Determine the maximum value from summed_preds to set the y-axis limit
    max_pred_value = np.max(summed_preds)

    # Create traces for the middle dimension of year_preds
    traces = []
    for i in range(summed_preds.shape[0]):  # Iterate over the 20 middle dimension
        trace = go.Scatter(
            x=np.arange(summed_preds.shape[1]),  # X-axis: 0 to 50
            y=summed_preds[i, :],  # Y-axis: summed predictions for each middle dimension
            mode='lines',
            name=f'Preds Trace {i+1}',
            visible=(i == 0)  # Only the first trace is visible initially
        )
        traces.append(trace)

    # Create bars for year_actuals
    actuals_trace = go.Bar(
        x=np.arange(summed_actuals.shape[0]),  # X-axis: 0 to 50
        y=summed_actuals,  # Y-axis: summed actuals
        name='Actuals',
        opacity=0.6
    )

    # Add the actuals trace to the list of traces
    traces.append(actuals_trace) 

    # Create the figure
    fig = go.Figure(data=traces)

    # Create steps for the slider
    steps = []
    for i in range(len(traces) - 1):  # Exclude the actuals trace from the slider
        step = dict(
            method='update',
            args=[{'visible': [j == i for j in range(len(traces) - 1)] + [True]}],  # Show only one preds trace and the actuals
            label=f'Preds Trace {i+1}'
        )
        steps.append(step)

    # Create the slider
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Trace: "},
        pad={"t": summed_preds.shape[1]},
        steps=steps
    )]

    # Update layout
    fig.update_layout(
        title='Predictions and Actuals',
        xaxis_title='Time Step',
        yaxis_title='Values',
        barmode='overlay',
        sliders=sliders,
        yaxis=dict(range=[0, max_pred_value])  # Set the y-axis scale based on the max of preds
    )

    return fig

def graph_harvest_cumsum(preds, actuals):
    # Sum preds by axis 0, then calculate cumulative sums by axis 1
    summed_preds = np.sum(preds, axis=0)
    cumsum_preds = np.cumsum(summed_preds, axis=1)
    
    # Calculate cumulative sums for actuals
    summed_actuals = np.sum(actuals, axis=0)
    cumsum_actuals = np.cumsum(summed_actuals)

    # Create traces
    traces = []
    for i in range(cumsum_preds.shape[0]):
        trace = go.Scatter(
            x=np.arange(cumsum_preds.shape[1]),
            y=cumsum_preds[i, :],
            mode='lines',
            line=dict(dash='dash'),
            name=f'Preds Trace {i+1}',
            visible=(i == 0)
        )
        traces.append(trace)

    # Add actuals trace
    actuals_trace = go.Scatter(
        x=np.arange(cumsum_actuals.shape[0]),
        y=cumsum_actuals,
        mode='lines',
        line=dict(color='red', dash='solid'),
        name='Actuals',
        visible = True
    )
    traces.append(actuals_trace)

    # Create steps for the slider
    steps = []
    for i in range(len(traces) - 1):  # Exclude the actuals trace from the slider
        step = dict(
            method='update',
            args=[{'visible': [j == i for j in range(len(traces) - 1)] + [True]}],  # Show only one preds trace and the actuals
            label=f'Preds Trace {i+1}'
        )
        steps.append(step)

    # Create the slider
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Trace: "},
        pad={"t": 20},
        steps=steps
    )]

    # Create figure
    fig = go.Figure(data=traces)

    # Update layout
    fig.update_layout(
        title='Cumulative Sum of Predictions and Actuals',
        xaxis_title='Time Step',
        yaxis_title='Cumulative Values',
        sliders=sliders
    )

    return fig

def graph_harvest_stacked(preds,actuals,labels):
    max_pred_value = np.max(preds)
    num_types, num_preds, num_weeks = preds.shape
    traces = []

    for i in range(num_preds):  # Iterate over the 20 middle dimension
        for j in range(num_types):
            trace = go.Scatter(
                x=np.arange(num_weeks),  # X-axis: 0 to 50
                y=preds[j,i, :],  # Y-axis: summed predictions for each middle dimension
                mode='lines',
                name=f'{labels[j]} Trace {i+1}',
                visible=(i == 0),  # Only the first trace is visible initially
                stackgroup=f'group{i}'
            )
            traces.append(trace)
    for j in range(num_types):
        actuals_trace = go.Bar(
                x=np.arange(num_weeks),  # X-axis: 0 to 50
                y=actuals[j],  # Y-axis: summed actuals
            name=f'{labels[j]} Actuals',
            opacity=0.6
        )
        traces.append(actuals_trace)

    fig = go.Figure(data=traces)
    
    steps = []
    for i in range(num_preds):  # Exclude the actuals trace from the slider
        step = dict(
            method='update',
            args=[{'visible': np.repeat([j == i for j in range(num_preds)]  + [True],num_types)}],  # Show only one preds trace and the actuals
            label=f'Preds Trace {i+1}'
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Trace: "},
        pad={"t": num_preds},
        steps=steps,
        len=1.0
    )]

    fig.update_layout(
        title='Predictions and Actuals',
        xaxis_title='Time Step',
        yaxis_title='Values',
        barmode='stack',
        sliders=sliders,
        yaxis=dict(range=[0, max_pred_value])  # Set the y-axis scale based on the max of preds
    )
    return fig