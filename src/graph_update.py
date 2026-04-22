import numpy as np

def sliding_transplant(fig,by_t_week,kg_by_transplant_week,week):

    fig.update_traces(
        y = kg_by_transplant_week.loc[week],
        selector = {'name': 'transplant_week_kg'},
    )

    fig.update_layout(
        shapes=[{
            'type': 'rect',
            'xref': 'x2',
            'yref': 'y2',
            'x0': week - 0.45,
            'x1': week + 0.45,
            'y0': 0,
            'y1': by_t_week.loc[week]['ha'],
            'fillcolor': 'black',
            'opacity': 0.2,
            'layer': 'above'
        }]
    )
    return fig

def sliding_transplant_projected(fig,data,week):
    fig.update_traces(
        y = data['in_CI'][week],
        selector = {'name': 'Inner Interval'}
    )

    fig.update_traces(
        y = data['out_CI'][week],
        selector = {'name': 'Outer Interval'}
    )

    fig.update_traces(
        y = data['actuals'][:week+1],
        x = data['szn_x'][:week+1],
        selector = {'name': 'Actuals'}
    )

    fig.update_traces(
        y = data['actuals'][week+1:],
        x = data['szn_x'][week+1:],
        selector = {'name': 'Later Actuals'}
    )

    fig.update_traces(
        y = data['mean_preds'][week],
        selector = {'name': 'Predicted'}
    )

    fig.update_layout(
        shapes=[
            {
                'type': 'line',
                'x0': week,
                'y0': 0,
                'x1': week,
                'y1': max(data['actuals']),
                'line': {
                    'color': "black",
                    'width': 2,
                    'dash': "dot"
                }
            }
        ]
    )
    return fig

def kg_ha_projected(fig,data,week):
    kg_so_far = data['actuals'][:,:week].sum(axis=1)
    rkg = data['preds'][:,week]
    total_kg = kg_so_far + rkg

    total_yield = total_kg.sum() / data['ha'].sum()
    max_kg = total_kg.max()
    max_line_ha = max_kg / total_yield

    kg_pairs_nan = np.array([[kg_so_far[i], total_kg[i], np.nan] for i in range(len(kg_so_far))]).flatten()

    fig.update_traces(
        y = kg_so_far,
        marker = {'size': (kg_so_far/max_kg*20 + 5)},
        selector = {'name': 'Total So Far'}
    )

    fig.update_traces(
        y = total_kg,
        marker = {'size': (total_kg/max_kg*20 + 5)},
        selector = {'name': 'Projected Total'}
    )

    fig.update_traces(
        y = kg_pairs_nan,
        selector = {'name': 'Connectors'}
    )

    fig.update_layout(
        shapes=[
            {
                "type": "line",
                "x0": 0,
                "x1": max_line_ha,
                "y0": 0,
                "y1": max_kg,
                "line": {
                    "color": "red",
                    "width": 2,
                    "dash": "dash"
                }
            }
        ],
        annotations=[
            {
                "x": max_line_ha,
                "y": max_kg,
                "text": f"Projected Yield: {total_yield/1000:.1f}k kg/ha",
                "showarrow": False,
                "yanchor": "bottom",
                "xanchor": "right",
                "font": {"color": "red"},
                "bgcolor": "white",
                "bordercolor": "red",
                "borderpad": 3
            }
    
        ]
    )
    return fig