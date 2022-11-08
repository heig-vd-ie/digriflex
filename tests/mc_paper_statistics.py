import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

if __name__ == '__main__':
    pio.kaleido.scope.mathjax = None
    case = ['Benchmark 1', 'Benchmark 2', 'Markov Chain']

    x0 = [0.036, 4.487, 6.815, 10.206, 3.559, 0.826, 0.446, 0.057, 2.504, 6.739, 7.008, 3.605]
    x1 = [6.860, 6.772, 10.223, 3.792, 0.096, 3.947, 0.350, 10.028, 3.372, 3.679, 0.401, 16.722]
    x2 = [3.363, 0.215, 4.013, 3.519, 4.179, 8.774, 1.405, 4.002, 7.036, 0.157, 0.204, 0.008]

    y0 = [3.104, 18.999, 24.542, 40.625, 26.139, 30.309, 16.583, 29.857, 7.210, 4.075, -0.310, -0.049]
    y1 = [39.791, 24.131, 55.111, 53.409, 57.281, 65.371, 53.128, 52.999, 44.173, 27.714, 42.995, -0.718]
    y2 = [36.980, -0.141, 6.254, 18.271, 31.022, 18.396, 15.924, 26.824, 0.000, -0.005, 34.919, -0.090]

    df0 = pd.DataFrame(np.array([x0, y0, [case[0]]*12]).T, columns=["Deviation", "Objective", "Case"])
    df1 = pd.DataFrame(np.array([x1, y1, [case[1]]*12]).T, columns=["Deviation", "Objective", "Case"])
    df2 = pd.DataFrame(np.array([x2, y2, [case[2]]*12]).T, columns=["Deviation", "Objective", "Case"])

    df = pd.concat([df0, df1, df2])
    df["Deviation"] = df["Deviation"].astype("float64")
    df["Objective"] = df["Objective"].astype("float64")

    # fig = px.scatter(df, x="Deviation", y="Objective", color="Case", marginal_x="box", marginal_y="box")
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Box(x=df["Case"].tolist(), y=df["Objective"].tolist(), fillcolor="darkgrey", marker_color="black"),
                  row=1, col=1)
    fig.add_trace(go.Box(x=df["Case"].tolist(), y=df["Deviation"].tolist(), fillcolor="darkgrey", marker_color="black"),
                  row=1, col=2)
    fig['layout']['xaxis']['title'] = ''
    fig['layout']['xaxis2']['title'] = ''
    fig['layout']['yaxis']['title'] = 'Objective ($)'
    fig['layout']['yaxis2']['title'] = 'Deviation (%)'
    fig['layout']["template"] = "ggplot2"
    fig['layout']["font"] = {
        "family": "Nunito",
        "size": 22,
    }
    fig['layout']["width"] = 1500
    fig['layout']["height"] = 500
    fig['layout']["showlegend"] = False
    plotly.io.write_image(fig, ".cache/figures/fig_statistics_analysis1.pdf", format="pdf")
    fig.show()
