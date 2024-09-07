import csv
import numpy as np

if __name__ == "__main__":
    """ Generates a LaTeX table of average hypervolume scores """

    print("pymoo & ParMOO-8 & ParMOO-16 & ParMOO-32\\\\\n\\hline")
    hvs = np.zeros(5)
    with open("../pymoo-dtlz2-results-2024/hv.csv", "r") as fp:
        csvreader = csv.reader(fp)
        for i, row in enumerate(csvreader):
            hvs[i] = float(row[0])
        print(f"{np.mean(hvs[i]):.2f}", end="")
    for SIZE in [8, 16, 32]:
        hvs[:] = 0
        for j, SEED in enumerate(range(5)):
            fname = f"../parmoo-dtlz2-results-2024/size{SIZE}_seed{SEED}.csv"
            with open(fname, "r") as fp:
                csvreader = csv.reader(fp)
                for row in csvreader:
                    hvs[j] += float(row[1])
                hvs[j] /= 4  # 4 rows for 4 different nthreads
        print(f"\t&\t{np.mean(hvs):.2f}", end="")
    print("\t\\\\")
