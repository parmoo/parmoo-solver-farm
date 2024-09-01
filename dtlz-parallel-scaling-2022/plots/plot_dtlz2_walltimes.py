import csv
import numpy as np
from plotly import graph_objects as go

if __name__ == "__main__":
    """ Generates the plot of average walltimes """

    nthreads = np.array([1, 2, 4, 8])
    sizes = [8, 16, 32]
    colors = ["blue", "green", "red"]
    walltimes = []
    for SIZE in sizes:
        timei = np.zeros(4)
        for SEED in range(5):
            fname = f"../parmoo-dtlz2-results-2024/size{SIZE}_seed{SEED}.csv"
            with open(fname, "r") as fp:
                csvreader = csv.reader(fp)
                for j, row in enumerate(csvreader):
                    timei[j] += float(row[0])
        timei[:] /= 5
        walltimes.append(timei)

    # Generate plotly graphs of results
    fig = go.Figure()
    # Add performance lines
    for i in range(3):
        fig.add_trace(go.Scatter(x=nthreads, y=walltimes[i], mode='lines',
                                 name=f"ParMOO-{sizes[i]}",
                                 line=dict(color=colors[i], width=2),
                                 showlegend=True))
    # Set the figure style/layout
    fig.update_layout(
        xaxis=dict(title="number of threads",
                   showline=True,
                   showgrid=True,
                   showticklabels=True,
                   linecolor='rgb(204, 204, 204)',
                   linewidth=2,
                   ticks='outside',
                   tickfont=dict(family='Arial', size=12)),
        yaxis=dict(title="walltime (seconds)",
                   showline=True,
                   showgrid=True,
                   showticklabels=True,
                   linecolor='rgb(204, 204, 204)',
                   linewidth=2,
                   ticks='outside',
                   tickfont=dict(family='Arial', size=12)),
        plot_bgcolor='white', width=500, height=300,
        margin=dict(l=80, r=50, t=20, b=20))
    
    # Save image
    fig.write_image("dtlz2_walltimes.eps")
    import time
    time.sleep(2)
    fig.write_image("dtlz2_walltimes.eps")
