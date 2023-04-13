import csv
import numpy as np
import plotly.graph_objects as go

F1 = "../results/fayans_structured_results_seed_"
F2 = "../results/fayans_blackbox_results_seed_"

# Generate filenames for structured solvers
fnames = []
for i in range(5):
    fnames.append(F1 + str(i) + ".csv")
# Read datasets and generate list of summary stats
x_vals1 = []
y_vals1 = []
for fname in fnames:
    print(f"Parsing file {fname}:")
    pts = []
    # Read in data
    try:
        print(f"  Reading {fname}...")
        with open(fname, "r") as fp:
            csv_reader = csv.reader(fp, delimiter=",")
            for i, row in enumerate(csv_reader):
                if i > 0:
                    pts.append([float(ci) for ci in row[-3:]])
        print(f"  Parsing {fname}...")
        # Add summary stats
        x_vals1.append([])
        y_vals1.append([])
        for i in range(2000, 10001, 10):
            y_vals1[-1].append(min([sum(fi) for fi in pts[:i]]))
            x_vals1[-1].append((i-2000)/10)
        print("  Done.")
    except FileNotFoundError:
        print(f"  Could not open {fname}. Skipping...")

# Generate filenames for unstructured solvers
fnames = []
for i in range(5):
    fnames.append(F2 + str(i) + ".csv")
# Read datasets and generate list of summary stats
x_vals2 = []
y_vals2 = []
for fname in fnames:
    print(f"Parsing file {fname}:")
    pts = []
    # Read in data
    try:
        print(f"  Reading {fname}...")
        with open(fname, "r") as fp:
            csv_reader = csv.reader(fp, delimiter=",")
            for i, row in enumerate(csv_reader):
                if i > 0:
                    pts.append([float(ci) for ci in row[-3:]])
        print(f"  Parsing {fname}...")
        # Add summary stats
        x_vals2.append([])
        y_vals2.append([])
        for i in range(2000, 10001, 10):
            y_vals2[-1].append(min([sum(fi) for fi in pts[:i]]))
            x_vals2[-1].append((i-2000)/10)
        print("  Done.")
    except FileNotFoundError:
        print(f"  Could not open {fname}. Skipping...")

# Generate plotly graphs of results
fig = go.Figure()
# Add performance lines
x_avg1 = np.zeros(len(x_vals1[0]))
y_avg1 = np.zeros(len(y_vals1[0]))
for i, (xi, yi) in enumerate(zip(x_vals1, y_vals1)):
    x_avg1[:] = (x_avg1[:] * i + xi[:])/ (i + 1)
    y_avg1[:] = (y_avg1[:] * i + yi[:])/ (i + 1)
fig.add_trace(go.Scatter(x=x_avg1, y=y_avg1, mode='lines',
                         name="structured",
                         line=dict(color="blue", width=2),
                         showlegend=True))
x_avg2 = np.zeros(len(x_vals2[0]))
y_avg2 = np.zeros(len(y_vals2[0]))
for i, (xi, yi) in enumerate(zip(x_vals2, y_vals2)):
    x_avg2[:] = (x_avg2[:] * i + xi[:])/ (i + 1)
    y_avg2[:] = (y_avg2[:] * i + yi[:])/ (i + 1)
fig.add_trace(go.Scatter(x=x_avg2, y=y_avg2, mode='lines',
                         name="blackbox",
                         line=dict(color="red", width=2),
                         showlegend=True))
# Set the figure style/layout
fig.update_layout(
    xaxis=dict(title="iteration",
               showline=True,
               showgrid=True,
               showticklabels=True,
               linecolor='rgb(204, 204, 204)',
               linewidth=2,
               ticks='outside',
               tickfont=dict(family='Arial', size=12)),
    yaxis=dict(title="$\chi^{2}$ loss",
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
fig.write_image("fayans_chisquared_convergence.eps")
import time
time.sleep(2)
fig.write_image("fayans_chisquared_convergence.eps")
