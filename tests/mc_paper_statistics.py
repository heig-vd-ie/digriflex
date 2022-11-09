from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.io as pio

if __name__ == '__main__':
    pio.kaleido.scope.mathjax = None
    case = ['NoClustering', 'Benchmark 1', 'Benchmark 2', 'Markov Chain']
    x0 = [6.670, 10.236, 6.743, 7.271, 3.463, 11.389, 3.473, 13.681, 0.070, 0.165, 3.360, 0.102]
    x1 = [3.737, 6.698, 6.951, 10.265, 3.860, 2.383, 4.131, 6.735, 0.321, 0.048, 3.416, 10.050]
    x2 = [3.815, 16.707, 10.046, 13.630, 0.076, 7.158, 0.628, 7.361, 3.382, 6.711, 20.107, 13.359]
    x3 = [0.165, 3.591, 4.026, 7.037, 3.415, 0.241, 0.067, 4.336, 3.433, 0.324, 3.044, 6.679]

    y0 = [36.343, -0.306, 9.400, 12.831, 32.390, 12.100, 14.173, 27.013, -0.003, -0.623, 34.072, -0.318]
    y1 = [3.104, 18.999, 24.542, 40.625, 26.139, 30.309, 16.583, 29.857, 7.210, 4.075, -0.310, -0.049]
    y2 = [39.987, 24.823, 55.932, 52.527, 57.004, 64.421, 54.397, 52.557, 44.467, 28.311, 43.442, -0.098]
    y3 = [40.507, 17.598, 22.575, 56.025, 40.264, 26.875, 27.551, 27.500, -0.094, 11.096, -0.056, -0.063]

    df0 = pd.DataFrame(np.array([x0, y0, [case[0]] * 12]).T, columns=["Deviation", "Objective", "Case"])
    df1 = pd.DataFrame(np.array([x1, y1, [case[1]] * 12]).T, columns=["Deviation", "Objective", "Case"])
    df2 = pd.DataFrame(np.array([x2, y2, [case[2]] * 12]).T, columns=["Deviation", "Objective", "Case"])
    df3 = pd.DataFrame(np.array([x3, y3, [case[3]] * 12]).T, columns=["Deviation", "Objective", "Case"])

    df = pd.concat([df0, df1, df2, df3])
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
