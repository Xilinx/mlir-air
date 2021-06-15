import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Plots paths')
parser.add_argument('-filename', default="", help='filename to be plotted')
parser.add_argument('-columnName', default="", help='Throughput or Latency')
args = parser.parse_args()

print(args.filename)

data = np.genfromtxt(args.filename, delimiter=' ', usecols=[0,1,2], names=True, dtype=int)

print(data['Area'])
values = data[args.columnName]

maxSoFar = 0
for i in range(0, len(values)):
    if values[i] < maxSoFar:
        values[i] = maxSoFar
    else:
        maxSoFar = values[i]

def plotTimes():
    plt.grid()
    plt.plot(data['Area'], data[args.columnName], label=args.columnName)
    plt.xticks(np.arange(0, 401, 10))
    plt.ylim(bottom=0)
    plt.legend()
    plt.ylabel("FPS")
    plt.show()

plotTimes()

