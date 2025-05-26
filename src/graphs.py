import plotly.graph_objects as go
from plotly.colors import sample_colorscale

def graph_season_table(table, title, palette='Agsunset'):
    figs = []
    summaries = []
    _,ranch_data,_ = table.graph_ready(True,False,False,include_actuals=True)
    summaries.append(table.summary(True,False,False))
    pf = ranch_data['preds_summed']
    af = ranch_data['actuals_summed']
    label = title + ' Ranches Predictions vs Actuals'
    figs.append(graph_with_actuals(pf,af,label,palette))


    _,class_data,_ = table.graph_ready(False,True,False,include_actuals=True)
    summaries.append(table.summary(False,True,False))
    pf = class_data['preds_summed']
    af = class_data['actuals_summed']
    label = title + ' Classes'
    figs.append(graph_with_actuals(pf,af,label,palette))

    _,type_data,_ = table.graph_ready(False,True,True,include_actuals=True)    
    summaries.append(table.summary(False,True,True))
    pf = type_data['preds_summed']
    af = type_data['actuals_summed']
    label = title + ' Types'
    figs.append(graph_with_actuals(pf,af,label,palette))

    _,ranch_class,_ = table.graph_ready(True,True,False,include_actuals=True)
    summaries.append(table.summary(True,True,False))
    pf = ranch_class['preds_summed']
    af = ranch_class['actuals_summed']
    label = title + ' Ranches, Classes, and Types'
    figs.append(graph_with_actuals(pf,af,label,palette))

    _,ranch_class_type,_ = table.graph_ready(True,True,True,include_actuals=True)
    summaries.append(table.summary(True,True,True))
    pf = ranch_class_type['preds_summed']
    af = ranch_class_type['actuals_summed']
    label = title + ' Ranches, Classes, and Types'
    figs.append(graph_with_actuals(pf,af,label,palette))

    return figs, summaries

def graph_transplant_table(table, title, palette='Agsunset'):
    figs = []
    summaries = []
    for key in ['summed','summed_cumsum','summed_cumprop']:
        _,ranch_data,_ = table.graph_ready(True,False,False,include_actuals=True)
        summaries.append(table.summary(True,False,False))
        pf = ranch_data['preds_' + key]
        af = ranch_data['actuals_' + key]
        label = title + ' Ranches '
        figs.append(graph_with_actuals(pf,af,label,palette))


        _,class_data,_ = table.graph_ready(False,True,False,include_actuals=True)
        summaries.append(table.summary(False,True,False))
        pf = class_data['preds_' + key]
        af = class_data['actuals_' + key]
        label = title + ' Classes ' + key
        figs.append(graph_with_actuals(pf,af,label,palette))

        _,type_data,_ = table.graph_ready(False,True,True,include_actuals=True)    
        summaries.append(table.summary(False,True,True))
        pf = type_data['preds_' + key]
        af = type_data['actuals_' + key]
        label = title + ' Types ' + key
        figs.append(graph_with_actuals(pf,af,label,palette))

        _,ranch_class,_ = table.graph_ready(True,True,False,include_actuals=True)
        summaries.append(table.summary(True,True,False))
        pf = ranch_class['preds_' + key]
        af = ranch_class['actuals_' + key]
        label = title + ' Ranches, Classes, and Types ' + key
        figs.append(graph_with_actuals(pf,af,label,palette))

        _,ranch_class_type,_ = table.graph_ready(True,True,True,include_actuals=True)
        summaries.append(table.summary(True,True,True))
        pf = ranch_class_type['preds_' + key]
        af = ranch_class_type['actuals_' + key]
        label = title + ' Ranches, Classes, and Types ' + key
        figs.append(graph_with_actuals(pf,af,label,palette))
    return figs, summaries

def graph_with_actuals(pf, af, title, palette='Agsunset'):
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
    x_values = pf.columns
    colors = sample_colorscale(palette, [i / (len(pf.index) - 1) for i in range(len(pf.index))])
    for i, row in enumerate(pf.index):
        label = str(row)
        fig.add_trace(go.Scatter(x=x_values, y=pf.loc[row], name=label, marker_color=colors[i]))
        fig.add_trace(go.Bar(x=x_values, y=af.loc[row], name=label, marker_color=colors[i], opacity=0.75))

    fig.update_layout(title=title)
    return fig

    

    
