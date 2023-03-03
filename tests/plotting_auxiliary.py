import numpy as np
import plotly.graph_objects as go
import math

def m_aep_func(y1, y2, x2, b):
    tmp = []
    if not b:
        for x1 in x2:
            xx = x1.hour * 6 + math.floor(x1.minute / 10)
            tmp.append(y1[xx])
        y1 = tmp
    if (y1 is not []) and (y2 is not []):
        err = np.average(np.abs(np.array(y1)- np.array(y2)))
    else:
        err = None
    return err

def figuring(x_arr, y, name, date_arr, y_lag, y_for, y_real):
    if name == 'DP':
        name1, name2, name3, name4 = 'DA Forecast', 'RT Forecast', '', 'Realized'
        yax_name = 'Demanded Active Power (kW)'
    elif name == 'DQ':
        name1, name2, name3, name4 = 'DA Forecast', 'RT Forecast', '', 'Realized'
        yax_name = 'Demanded Reactive Power (kVar)'
    elif name == 'PV':
        name1, name2, name3, name4 = 'DA Forecast', 'RT Forecast', 'Available', 'Deployed'
        yax_name = 'PV Production (kW)'
    elif name == 'FP':
        name1, name2, name3, name4 = 'DA Schedule', 'Asked in RT', '', 'Realized'
        yax_name = 'Connection Point Active Power (kW)'
    elif name == 'FQ':
        name1, name2, name3, name4 = 'DA Schedule', 'Asked in RT', '', 'Realized'
        yax_name = 'Connection Point Reactive Power (kVar)'
    elif name == 'SOC':
        name1, name2, name3, name4 = 'DA Schedule', '', '', 'RT Measurement'
        yax_name = 'State of Charge (%)'
    elif name == 'SP':
        name1, name2, name3, name4 = '', '', '', 'Set point'
        yax_name = 'kW or kVAR'
    else:
        name1, name2, name3, name4 = '', '', '', 'Set point'
        yax_name = 'kW or kVAR'
    if len(y_lag) == 0:
        date_arr1 = []
    else:
        date_arr1 = date_arr # - pd.DateOffset(hours=1)
    if len(y_for) == 0:
        date_arr2 = []
    else:
        date_arr2 = date_arr # - pd.DateOffset(hours=1)
    if len(y_real) == 0:
        date_arr3 = []
    else:
        date_arr3 = date_arr # - pd.DateOffset(hours=1)
    fig = go.Figure([])
    fig.update_layout(yaxis_title=yax_name, hovermode="x", width=700, height=400)
    fig.update_traces(mode='lines')
    fig.add_trace(go.Scatter(x=date_arr3, y=y_real, mode="lines", name=name4,
                             line=dict(color='firebrick', width=2.5)))
    fig.add_trace(go.Scatter(x=date_arr1, y=y_lag, mode="lines", name=name3,
                             line=dict(color='rgba(65, 105, 225, 0.8)', width=2)))
    fig.add_trace(go.Scatter(x=date_arr2, y=y_for, mode="lines", name=name2,
                             line=dict(color='royalblue', width=1.5, dash='dot')))
    fig.add_trace(go.Scatter(name='Upper Bound', x=x_arr, y=np.array(y[0, :]) + np.array(y[1, :]),
                             mode='lines', marker=dict(color='rgba(30, 70, 100, 0.3)'),
                             line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(name='Lower Bound', x=x_arr, y=np.array(y[0, :]) - np.array(y[2, :]),
                             marker=dict(color="#444"), line=dict(width=0),
                             mode='lines', fillcolor='rgba(30, 70, 100, 0.3)',
                             fill='tonexty', showlegend=False))
    fig.add_trace(go.Scatter(x=x_arr, y=np.array(y[0, :]), line_color='rgb(30, 70, 100)', name=name1))
    fig.show()
