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

def sliding_transplant():
    """
    Graph the kilos received and Ha planted for the whole season. 
    When a specific transplant week is selected, the graph will show the kilos received curve/yield for that week.
    """
    fig = go.Figure()

    #----------initialize traces----------

    #add total curve trace (named: total_kg)

    #add highlighted week curve trace (named: transplant_week_kg)

    #add ha bar graph trace (named: ha)

    #----------initialize shapes----------
    #add shape for highlighted week bar
    #add shape for first harvest
    #add shape for harvest duration
    #add shape for end of harvest
    #add shape for current week

    #----------initialize annotations----------
    #add annotation for total yield

    #add annotation for highlighted week yield

    #----------update layout----------

    return fig

def sliding_transplant_projected():
    """
    Graph the kilos projected and Ha planted for the whole season. 
    When a specific transplant week is selected, the graph will show the kilos projected curve/yield for that week.
    """
    fig = go.Figure()

    #----------initialize traces----------

    #add total curve trace (named: total_kg)

    #add confidence interval curve traces (named: ci_kg)

    #add highlighted week curve trace (named: transplant_week_kg)

    #add ha bar graph trace (named: ha)

    #----------initialize shapes----------
    #add shape for highlighted week bar
    #add shape for first harvest
    #add shape for harvest duration
    #add shape for end of harvest
    #add shape for current week

    #----------initialize annotations----------
    #add annotation for total yield

    #add annotation for highlighted week yield

    #----------update layout----------

    return fig

def sliding_transplant_ranch():
    """
    Graph the kilos received and Ha planted for the whole season with a ranch breakdown. 
    When a specific transplant week is selected, the graph will show the kilos received curve/yield for that week with a ranch breakdown.
    """
    fig = go.Figure()

    #----------initialize traces----------
    #add total curve trace (named: total_kg)

    #add highlighted week curve traces for each ranch(named: [ranch]_transplant_week_kg, legendgroup = [ranch])

    #add ha bar graph traces for each ranch(named: [ranch]_ha, legendgroup = [ranch])

    #add highlighted week bar trace (named: highlighted_week

    #----------initialize shapes----------
    #add shape for first harvest
    #add shape for harvest duration
    #add shape for end of harvest
    #add shape for current week

    #----------initialize annotations----------
    #add annotation for total yield

    #add annotation for highlighted week yield

    #----------update layout----------

    return fig

def sliding_transplant_ranch_projected():
    """
    Graph the kilos received and Ha planted for the whole season with a ranch breakdown. 
    When a specific transplant week is selected, the graph will show the kilos received curve/yield for that week with a ranch breakdown.
    """
    fig = go.Figure()

    #----------initialize traces----------
    #add total curve trace (named: total_kg)

    #add highlighted week curve traces for each ranch(named: [ranch]_transplant_week_kg)

    #add ha bar graph traces for each ranch(named: [ranch]_ha)

    #add highlighted week bar trace (named: highlighted_week)

    #----------initialize shapes----------
    #add shape for first harvest

    #add shape for harvest duration

    #add shape for end of harvest

    #add shape for current week

    #----------initialize annotations----------
    #add annotation for total yield

    #add annotation for highlighted week yield

    #----------update layout----------

    return fig

def kg_ha():
    """
    Graphs the kg vs ha for each batch. Also shows the total yield.
    """
    fig = go.Figure()

    #----------initialize traces----------
    #add kg vs ha scatter traces, bigger circles for more kg, more yellow for older years, green for newer years (named: [year]_kg)

    #----------initialize shapes----------
    #add shape for total yield line

    #----------initialize annotations----------
    #add annotation for total yield

    #----------update layout----------

    return fig

def kg_ha_projected():
    """
    Graphs the actual and projected kg vs ha for each batch. Also shows the total yield.
    """
    fig = go.Figure()

    #----------initialize traces----------
    #add kg vs ha actual so far traces, bigger circles for more kg (named: kg_so_far)

    #add kg vs ha projected traces, bigger circles for more kg (named: projected_kg)

    #add lines between actual and projected traces with same opacity (named: connectors)

    #----------initialize shapes----------
    #add shape for total projectedyield line

    #add shape for historical total yield line

    #----------initialize annotations----------
    #add annotation for total projected yield

    #add annotation for historical total yield

    #----------update layout----------

    return fig

def transplant_week_yield():
    """
    Graphs the yield for each week over the course of the season.
    """

    fig = go.Figure()

    #----------initialize traces----------
    #add yield vs transplant week traces, bigger circles for more kg, more yellow for older years, green for newer years (named: [year]_yield)

    #add solid red? line of average yield for each transplant week (named: average_yield)

    #----------initialize shapes----------
    #add dashed line of overall average yield

    #----------initialize annotations----------
    #add annotation for overall average yield


    #----------update layout----------

    return fig

def transplant_week_yield_projected():
    """
    Graphs the actual and projected yield for each week over the course of the season.
    """

    fig = go.Figure()

    #----------initialize traces----------
    #add yield so far vs transplant week traces, bigger circles for more kg. scatter before reg x value (minus random float) (named: yield_so_far)

    #add projected yield vs transplant week traces, bigger circles for more kg, scatter after reg x value (plus random float) (named: projected_yield)

    #add lines between actual and projected traces with same opacity (named: connectors)

    # add a line for projected average yield for each transplant week (named: projected_average_yield)

    # add a line for actual so far average yield for each transplant week (named: average_yield_so_far)

    #----------initialize shapes----------

    #add shape for historical total yield line

    #----------initialize annotations----------
    #add annotation for total projected yield

    #add annotation for historical total yield

    #----------update layout----------

    return fig

def ranch_breakdown():
    """
    Graph yield for each ranch
    """
    fig = go.Figure()
    #----------initialize traces----------
    #yield vs ranch traces for each year (named: [year]_yield)

    #add a bar for average yield for each ranch (named: average_yield)

    #---------- initialize shapes----------
    #add a shape for total yield line

    #---------- initialize annotations----------
    #add an annotation for total yield

    #---------- update layout----------

    return fig

def ranch_breakdown_projected():
    """
    Graph projected yield for each ranch
    """
    fig = go.Figure()

    #----------initialize traces----------
    #yield so far vs ranch traces for each year, kilo scaled (named: yield_so_far)

    #projected yield vs ranch traces for each year, kilo scaled (named: projected_yield)
    
    #add a bar for average historical yield for each ranch (named: average_historical_yield)

    #add a bar for average projected yield for each ranch (named: average_projected_yield)

    #---------- initialize shapes----------
    #add a shape for total yield line

    #---------- initialize annotations----------
    #add an annotation for total yield

    #---------- update layout----------

    return fig

def yield_breakdown():
    """
    Graph yield breakdown over the course of each year
    """
    fig = go.Figure()
    #----------initialize traces----------
    #historical yield for each class (named: [class]_historical_yield)

    #projected yield, dotted line from end of historical yield for each class (named: [class]_projected_yield)


    #---------- initialize shapes/annotations----------
    #maybe shape of average yield for each one / annotation? but might be to cluttered

    #---------- update layout----------
    return fig
