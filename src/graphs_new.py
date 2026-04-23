import plotly.graph_objects as go
import plotly.colors as pc
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

def get_gradient(name,n):
    for module in (pc.sequential, pc.diverging, pc.cyclical):
        if hasattr(module, name):
            color_scale = getattr(module, name)
            indices = np.linspace(0, len(color_scale)-1, n, dtype=int)
            colors = [color_scale[i] for i in indices]
            return colors
    raise ValueError(f"Colorscale '{name}' not found in Plotly colors.")

def sliding_transplant(data):
    """
    Graph the kilos received and Ha planted for the whole season. 
    When a specific transplant week is selected, the graph will show the kilos received curve/yield for that week.
    """

    fig = make_subplots(rows=2, cols=1)

    #----------initialize traces----------

    #add total curve trace to subplot 0 (named: total_kg)
    fig.add_trace(
        go.Scatter(
            x = data['szn_x'],
            y = data['total_kg'],
            mode = 'lines',
            name = 'total_kg'
        ),
        row = 1,
        col = 1
    )

    #add highlighted week curve trace to subplot 0 (named: transplant_week_kg)
    fig.add_trace(
        go.Bar(
            x = data['szn_x'],
            y = data['initial_transplant_y'],
            name = 'transplant_week_kg'
        )
    )

    #add ha bar graph trace to subplot 1 (named: ha)
    fig.add_trace(
        go.Bar(
            x = data['szn_transplant_weeks'],
            y = data['szn_ha'],
            name = 'ha'
        ),
        row = 2,
        col = 1
    )

    # dummy trace for scaling
    fig.add_trace(
        go.Bar(
            x = data['szn_x'],
            y = np.zeros(len(data['szn_x'])),
            showlegend=False,
            name = 'zeros'
        ),
        row = 2,
        col = 1)
    #----------initialize shapes----------s
    #add shape for highlighted week bar to subplot 1
    fig.add_shape(
        type="rect",
        xref="x2",
        yref="y2",
        x0= data['szn_transplant_weeks'][0] - 0.45,
        x1= data['szn_transplant_weeks'][0] + 0.45,
        y0=0,
        y1=data['szn_ha'][0],
        fillcolor='black',
        opacity=0.2,
        layer="above",
    )
    #add shape for first harvest to subplot 0
    #add shape for harvest duration to subplot 0
    #add shape for end of harvest to subplot 0
    #add shape for current week to subplot 1

    #----------initialize annotations----------
    #add annotation for total yield to subplot 0

    #add annotation for highlighted week yield to subplot 0



    #----------update layout----------
    fig.update_layout(
        title='Kg vs Week of the Season',
        yaxis={'title':'Kg'},
        xaxis={'title':'Week of the Season'},
        yaxis2={'title':'Ha'},
        barmode='overlay'
    )

    return fig

def sliding_transplant_projected(data):
    """
    Graph the kilos projected and Ha planted for the whole season. 
    When a specific transplant week is selected, the graph will show the kilos projected curve/yield for that week.
    """
    fig = go.Figure()

    #----------initialize traces----------

    #add total curve trace (named: total_kg)
    fig.add_trace(go.Scatter(
        x=data['szn_x'],
        y=data['mean_preds'][0],
        mode='lines',
        name='total_kg', 
        line={'color': 'red', 'dash': 'dash'})
    )
    #add confidence interval curve traces (named: ci_kg)
    fig.add_trace(go.Scatter(
        x=data['interval_x'],
        y=data['in_CI'][0],
        fill='toself',
        fillcolor='rgba(255, 105, 180, 0.3)',
        line={'color': 'rgba(255, 105, 180, 0)'},
        hoverinfo='skip',
        showlegend=True,
        name='Inner Interval'
    ))

    fig.add_trace(go.Scatter(
        x=data['interval_x'],
        y=data['out_CI'][0],
        fill='toself',
        fillcolor='rgba(255, 182, 193, 0.3)',
        line={'color': 'rgba(255, 182, 193, 0)'},
        hoverinfo='skip',
        showlegend=True,
        name='Outer Interval'
    ))

    #add actual trace (named: actual_kg)
    fig.add_trace(go.Bar(x=data['szn_x'][1:], y=data['actuals'][1:], marker_color='black',name='Later Actuals',legendgroup='actuals',opacity=0.5,showlegend=False))
    fig.add_trace(go.Bar(x=data['szn_x'][:1], y=data['actuals'][:1], marker_color='black',name='Actuals',legendgroup='actuals'))

    #add highlighted week curve trace (named: transplant_week_kg)

    #add ha bar graph trace (named: ha)

    #----------initialize shapes----------
    #add shape for highlighted week bar
    #add shape for first harvest
    #add shape for harvest duration
    #add shape for end of harvest
    #add shape for current week
    fig.add_vline(x=0, line_width=2, line_dash="dot", line_color="black", annotation_text="Current", annotation_position="top right")

    #----------initialize annotations----------
    #add annotation for total yield

    #add annotation for highlighted week yield

    #----------update layout----------

    fig.update_layout(
    title='Projected KG',
    yaxis_title='KG',
    xaxis_title='Week of Season'
    )

    return fig

# def sliding_transplant_ranch():
#     """
#     Graph the kilos received and Ha planted for the whole season with a ranch breakdown. 
#     When a specific transplant week is selected, the graph will show the kilos received curve/yield for that week with a ranch breakdown.
#     """
#     fig = go.Figure()

#     #----------initialize traces----------
#     #add total curve trace (named: total_kg)

#     #add highlighted week curve traces for each ranch(named: [ranch]_transplant_week_kg, legendgroup = [ranch])

#     #add ha bar graph traces for each ranch(named: [ranch]_ha, legendgroup = [ranch])

#     #add highlighted week bar trace (named: highlighted_week

#     #----------initialize shapes----------
#     #add shape for first harvest
#     #add shape for harvest duration
#     #add shape for end of harvest
#     #add shape for current week

#     #----------initialize annotations----------
#     #add annotation for total yield

#     #add annotation for highlighted week yield

#     #----------update layout----------

#     return fig

# def sliding_transplant_ranch_projected():
#     """
#     Graph the kilos received and Ha planted for the whole season with a ranch breakdown. 
#     When a specific transplant week is selected, the graph will show the kilos received curve/yield for that week with a ranch breakdown.
#     """
#     fig = go.Figure()

#     #----------initialize traces----------
#     #add total curve trace (named: total_kg)

#     #add highlighted week curve traces for each ranch(named: [ranch]_transplant_week_kg)

#     #add ha bar graph traces for each ranch(named: [ranch]_ha)

#     #add highlighted week bar trace (named: highlighted_week)

#     #----------initialize shapes----------
#     #add shape for first harvest

#     #add shape for harvest duration

#     #add shape for end of harvest

#     #add shape for current week

#     #----------initialize annotations----------
#     #add annotation for total yield

#     #add annotation for highlighted week yield

#     #----------update layout----------

#     return fig

def kg_ha(data):
    """
    Graphs the kg vs ha for each batch. Also shows the total yield.
    """
    fig = go.Figure()

    #----------initialize traces----------
    #add kg vs ha scatter traces, bigger circles for more kg, more yellow for older years, green for newer years (named: [year]_kg)
    for i, year in enumerate(data['years']):
        color = data['colors'][i]
        df = data['meta'][data['meta']['year'] == year]
        fig.add_trace(
            go.Scatter(
                x=df['ha'],
                y=df['total_kg'],
                mode='markers',
                name=str(year),
                marker=dict(
                    size=(df['total_kg']/data['max_kg']*20 + 5),
                    color=color,
                    opacity=0.7,
                    line=dict(width=1, color='black')
                ),
            ))
    #----------initialize shapes----------
    #add shape for total yield line
    fig.add_shape(
        type="line",
        x0=0,
        x1=data['max_line_ha'],
        y0=0,
        y1=data['max_kg'],
        line={
            "color": "red",
            "width": 2,
            "dash": "dash"
        },
    )

    #----------initialize annotations----------
    #add annotation for total yield
    fig.add_annotation(
        x=data['max_line_ha'],
        y=data['max_kg'],
        text=f"Total Yield: {data['total_yield']/1000:.1f}k kg/ha",
        showarrow=False,
        yanchor="bottom",
        xanchor="right",
        font={"color": "red"},
        bgcolor="white",
        bordercolor="red",
        borderpad=3
    )

    #----------update layout----------
    fig.update_layout(
        title='Kg vs Ha',
        xaxis_title='Ha',
        yaxis_title='Kg'
    )
    return fig

def kg_ha_projected(data):
    """
    Graphs the actual and projected kg vs ha for each batch. Also shows the total yield.
    """
    fig = go.Figure()
    kg_so_far = data['actuals'][:,:0].sum(axis=1)
    rkg = data['preds'][:,0]
    total_kg = kg_so_far + rkg
    ha = data['ha']

    total_yield = total_kg.sum() / ha.sum()
    max_kg = total_kg.max()
    max_line_ha = max_kg / total_yield

    kg_pairs_nan = np.array([[kg_so_far[i], total_kg[i], np.nan] for i in range(len(kg_so_far))]).flatten()
    ha_repeated = np.repeat(ha, 3)
    #----------initialize traces----------
    #add kg vs ha actual so far traces, bigger circles for more kg (named: kg_so_far)
    fig.add_trace(
        go.Scatter(
            x=ha,
            y=kg_so_far,
            mode='markers',
            name='Total So Far',
            marker={
                'size': (kg_so_far/total_kg.max()*20 + 5),
                'color': 'black',
                'opacity': 0.7
            }
        )
    )

    #add kg vs ha projected traces, bigger circles for more kg (named: projected_kg)
    fig.add_trace(
        go.Scatter(
            x=ha,
            y=total_kg,
            mode='markers',
            name='Projected Total',
            marker={
                'size': (total_kg/total_kg.max()*20 + 5),
                'color': 'green',
                'opacity': 0.7
            }
        )
    )
    #add lines between actual and projected traces with same opacity (named: connectors)
    fig.add_trace(
        go.Scatter(
            x=ha_repeated,
            y=kg_pairs_nan.flatten(),
            mode='lines',
            name='Connectors',
            line=dict(width=0.2,color='green',dash='dash')
    ))

    #----------initialize shapes----------
    #add shape for total projectedyield line
    fig.add_shape(
        type="line",
        x0=0,
        x1=max_line_ha,
        y0=0,
        y1=max_kg,
        line={
            "color": "red",
            "width": 2,
            "dash": "dash"
        },
    )

    #add shape for historical total yield line

    #----------initialize annotations----------
    #add annotation for total projected yield
    fig.add_annotation(
        x=max_line_ha,
        y=max_kg,
        text=f"Projected Yield: {total_yield/1000:.1f}k kg/ha",
        showarrow=False,
        yanchor="bottom",
        xanchor="right",
        font={"color": "red"},
        bgcolor="white",
        bordercolor="red",
        borderpad=3
    )
    #add annotation for historical total yield

    #----------update layout----------
    fig.update_layout(
        title='Kg vs Ha',
        xaxis_title='Ha',
        yaxis_title='Kg'
    )
    return fig

def transplant_week_yield(df):
    """
    Graphs the yield for each week over the course of the season.
    """

    fig = go.Figure()
    total_yield = df['total_kg'].sum() / df['ha'].sum()
    max_kg = df['total_kg'].max()

    df1 = df.groupby('transplant_week').agg({'total_kg': 'sum', 'ha': 'sum'})
    df1['yield'] = df1['total_kg'] / df1['ha']


    years = np.sort(df.year.unique())
    colors = get_gradient('YlGn',len(years))

    #----------initialize traces----------
    #add yield vs transplant week traces, bigger circles for more kg, more yellow for older years, green for newer years (named: [year]_yield)
    for i, year in enumerate(years):
        color = colors[i]
        yf = df[df.year == year]
        fig.add_trace(
            go.Scatter(
                x = yf['x'],
                y = yf['yield'],
                mode = 'markers',
                name = str(year),
                marker={
                    "size": (yf['total_kg']/max_kg*20 + 5),
                    "color": color,
                    "opacity": 0.7,
                    "line": {"width": 1, "color": color},
                },
            )
        )
    #add solid red? line of average yield for each transplant week (named: average_yield)
    fig.add_trace(
        go.Scatter(
            x=df1.index, 
            y=df1['yield'], 
            mode='lines+markers', 
            line = {'color': 'red', 'width': 2},
            name = 'Average Yield'
        )
    )
    
    #----------initialize shapes----------
    #add dashed line of overall average yield
    fig.add_hline(y=total_yield, line_dash="dash", line_color="black",annotation_text=f"Total Yield: {total_yield/1000:.1f}k kg/ha", annotation_position="bottom right")
    #----------initialize annotations----------
    #add annotation for overall average yield


    #----------update layout----------
    fig.update_layout(
        title='Yield vs Transplant Week',
        xaxis_title='Transplant Week',
        yaxis_title='Yield (kg/ha)'
    )
    return fig

def transplant_week_yield_projected(df, preds, actuals):
    """
    Graphs the actual and projected yield for each week over the course of the season.
    """

    fig = go.Figure()
    kg_so_far = actuals[:,:0].sum(axis=1)
    rkg = preds[:,0]
    total_kg = kg_so_far + rkg
    ha = df['ha'].values
    df['total_kg'] = total_kg

    average_yield_per_week = df.groupby('transplant_week')['total_kg'].sum() / df.groupby('transplant_week')['ha'].sum()

    yield_so_far = kg_so_far / ha
    predicted_yield = total_kg / ha

    total_predicted_yield = total_kg.sum() / ha.sum()

    kg_pairs_nan = np.array([[yield_so_far[i], predicted_yield[i], np.nan] for i in range(len(kg_so_far))]).flatten()
    x_pairs_nan = np.array([[df['x1'].iloc[i], df['x2'].iloc[i], np.nan] for i in range(len(df))]).flatten()

    #----------initialize traces----------
    #add yield so far vs transplant week traces, bigger circles for more kg. scatter before reg x value (minus random float) (named: yield_so_far)
    fig.add_trace(
        go.Scatter(
            x=df['x1'],
            y=yield_so_far,
            mode='markers',
            name='Yield So Far',
            marker={
                'size': (kg_so_far/total_kg.max()*20 + 5),
                'color': 'black',
                'opacity': 0.7
            }
        )
    )
    #add projected yield vs transplant week traces, bigger circles for more kg, scatter after reg x value (plus random float) (named: projected_yield)
    fig.add_trace(
        go.Scatter(
            x=df['x2'],
            y=predicted_yield,
            mode='markers',
            name='Projected Yield',
            marker={
                'size': (total_kg/total_kg.max()*20 + 5),
                'color': 'green',
                'opacity': 0.7
            }
        )
    )
    #add lines between actual and projected traces with same opacity (named: connectors)
    fig.add_trace(
        go.Scatter(
            x=x_pairs_nan,
            y=kg_pairs_nan.flatten(),
            mode='lines',
            name='Connectors',
            line=dict(width=0.2,color='green',dash='dash')
    ))
    # add a line for projected average yield for each transplant week (named: projected_average_yield)
    fig.add_trace(
        go.Scatter(
            x=average_yield_per_week.index,
            y=average_yield_per_week,
            mode='lines',
            name='Average Yield',
            line=dict(width=2,color='red',dash='dash')
        )
    )


    fig.add_hline(y=total_predicted_yield, line_dash="dash", line_color="black",annotation_text=f"Total Predicted Yield: {total_predicted_yield/1000:.1f}k kg/ha", annotation_position="bottom right")

    # add a line for actual so far average yield for each transplant week (named: average_yield_so_far)

    #----------initialize shapes----------

    #add shape for historical total yield line

    #----------initialize annotations----------
    #add annotation for total projected yield

    #add annotation for historical total yield

    #----------update layout----------
    fig.update_layout(
        title='Yield vs Transplant Week',
        xaxis_title='Transplant Week',
        yaxis_title='Yield (kg/ha)'
    )


    return fig

def ranch_yield(df,mappings):
    """
    Graph yield for each ranch
    """
    fig = go.Figure()

    df1 = df.groupby('ranch').agg({'total_kg': 'sum', 'ha': 'sum'})
    df1['yield'] = df1['total_kg'] / df1['ha']

    total_yield = df['total_kg'].sum() / df['ha'].sum()
    years = np.sort(df.year.unique())
    max_kg = df['total_kg'].max()
    colors = get_gradient('YlGn',len(years))
    #----------initialize traces----------
    #yield vs ranch traces for each year (named: [year]_yield)
    for i, year in enumerate(years):
        color = colors[i]
        yf = df[df.year == year]
        fig.add_trace(
            go.Scatter(
                x = yf['x'],
                y = yf['yield'],
                mode = 'markers',
                name = str(year),
                marker={
                    "size": (yf['total_kg']/max_kg*20 + 5),
                    "color": color,
                    "opacity": 0.7,
                    "line": {"width": 1, "color": color},
                },
            )
        )

    #add a bar for average yield for each ranch (named: average_yield)
    fig.add_trace(go.Bar(
        x=df1.index.map(mappings['ranch']),
        y=df1['yield'],  
        name = 'Average Yield',
        marker_color='pink',
        opacity=0.5))

    #---------- initialize shapes----------
    #add a shape for total yield line
    ### Change to be a transplant week line
    fig.add_hline(
        y=total_yield,
        line_dash="dash", 
        line_color="black",
        annotation_text=f"Total Yield: {total_yield/1000:.1f}k kg/ha",
        annotation_position="bottom right"
    )

    #---------- initialize annotations----------
    #add an annotation for total yield

    #---------- update layout----------
    fig.update_layout(
        title='Yield vs Ranch',
        xaxis_title='Ranch',
        yaxis_title='Yield (kg/ha)',
        xaxis = {
            "ticktext": list(mappings['ranch'].keys()),
            "tickvals": list(mappings['ranch'].values()),
        }
    )
    return fig

def ranch_yield_projected(df, preds, actuals, mappings):
    """
    Graph projected yield for each ranch
    """
    fig = go.Figure()

    kg_so_far = actuals[:,:0].sum(axis=1)
    rkg = preds[:,0]
    total_kg = kg_so_far + rkg
    ha = df['ha'].values
    df['total_kg'] = total_kg

    yield_so_far = kg_so_far / ha
    predicted_yield = total_kg / ha

    average_yield_per_ranch = df.groupby('ranch')['total_kg'].sum() / df.groupby('ranch')['ha'].sum()
    total_predicted_yield = total_kg.sum() / ha.sum()

    kg_pairs_nan = np.array([[yield_so_far[i], predicted_yield[i], np.nan] for i in range(len(kg_so_far))]).flatten()
    x_pairs_nan = np.array([[df['x1'].iloc[i], df['x2'].iloc[i], np.nan] for i in range(len(df))]).flatten()
    #----------initialize traces----------
    #yield so far vs ranch traces  kilo scaled (named: yield_so_far)
    fig.add_trace(
        go.Scatter(
            x=df['x1'],
            y=yield_so_far,
            mode='markers',
            name='Yield So Far',
            marker={
                'size': (kg_so_far/total_kg.max()*20 + 5),
                'color': 'black',
                'opacity': 0.7
            }
        )
    )
    #projected yield vs ranch traces for each year, kilo scaled (named: projected_yield)
    fig.add_trace(
        go.Scatter(
            x=df['x2'],
            y=predicted_yield,
            mode='markers',
            name='Projected Yield',
            marker={
                'size': (total_kg/total_kg.max()*20 + 5),
                'color': 'green',
                'opacity': 0.7
            }
        )
    )
    #add a bar for average historical yield for each ranch (named: average_historical_yield)
    # connectors
    fig.add_trace(
        go.Scatter(
            x=x_pairs_nan,
            y=kg_pairs_nan.flatten(),
            mode='lines',
            name='Connectors',
            line=dict(width=0.2,color='green',dash='dash')
    ))
    #add a bar for average projected yield for each ranch (named: average_projected_yield)


    fig.add_trace(
        go.Bar(
            x=average_yield_per_ranch.index.map(mappings['ranch']),
            y=average_yield_per_ranch,
            name='Projected Average Yield',
            marker_color='pink',
            opacity=0.5
        )
    )

    fig.add_hline(y=total_predicted_yield, line_dash="dash", line_color="black",annotation_text=f"Total Predicted Yield: {total_predicted_yield/1000:.1f}k kg/ha", annotation_position="bottom right")
    #---------- initialize shapes----------
    #add a shape for total yield line

    #---------- initialize annotations----------
    #add an annotation for total yield

    #---------- update layout----------
    fig.update_layout(
        title='Yield vs Ranch',
        xaxis_title='Ranch',
        yaxis_title='Yield (kg/ha)',
        xaxis = {
            "ticktext": list(mappings['ranch'].keys()),
            "tickvals": list(mappings['ranch'].values()),
        }
    )

    return fig

def yield_breakdown(df, df_by_year, preds, actuals,df1,historical_yield):
    """
    Graph yield breakdown over the course of each year
    """
    fig = go.Figure()

    max_kg = df['total_kg'].max()

    kg_so_far = actuals[:,:0].sum(axis=1)
    rkg = preds[:,0]
    total_kg = kg_so_far + rkg
    ha = df1['ha'].values

    yield_so_far = kg_so_far / ha
    predicted_yield = total_kg / ha

    total_predicted_yield = total_kg.sum() / ha.sum()

    kg_pairs_nan = np.array([[yield_so_far[i], predicted_yield[i], np.nan] for i in range(len(kg_so_far))]).flatten()
    x_pairs_nan = np.array([[df1['x1'].iloc[i], df1['x2'].iloc[i], np.nan] for i in range(len(df1))]).flatten()

 

    #----------initialize traces----------
    #historical yield for each class (named: [class]_historical_yield)
    fig.add_trace(
        go.Scatter(
            x=df['x'],
            y=df['yield'],
            mode='markers',
            name='Historical Yields',
            marker={
                'size': (df['total_kg']/max_kg*20 + 5),
                'color': 'black',
                'opacity': 0.5
            }
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df1['x1'],
            y=yield_so_far,
            mode='markers',
            name='Yield So Far',
            marker={
                'size': (kg_so_far/max_kg*20 + 5),
                'color': 'black',
                'opacity': 0.7
            }
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df1['x2'],
            y=predicted_yield,
            mode='markers',
            name='Projected Yield',
            marker={
                'size': (total_kg/max_kg*20 + 5),
                'color': 'green',
                'opacity': 0.7
            }
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_pairs_nan,
            y=kg_pairs_nan,
            mode='lines',
            name='Connectors',
            line=dict(width=0.2,color='green',dash='dash')
    ))
    #projected yield, dotted line from end of historical yield for each class (named: [class]_projected_yield)
    fig.add_trace(
        go.Scatter(
            x=df_by_year.index,
            y=df_by_year['yield'],
            name='Historical Average Yield',
            mode='lines',
            line=dict(width=2,color='red',dash='dash')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[df_by_year.index.values[-1],2024],
            y=[df_by_year['yield'].values[-1],total_predicted_yield],
            name='Projected Average Yield',
            mode='lines',
            line=dict(width=2,color='red',dash='dot')
        )
    )

    #---------- initialize shapes/annotations----------
    #maybe shape of average yield for each one / annotation? but might be to cluttered
    fig.add_hline(y=historical_yield, line_dash="dash", line_color="orange",annotation_text=f"Historical Yield: {historical_yield/1000:.1f}k kg/ha", annotation_position="bottom right")
    #---------- update layout----------
    
    fig.update_layout(
        title='Yield vs Year',
        xaxis_title='Year',
        yaxis_title='Yield (kg/ha)'
    )

    return fig
