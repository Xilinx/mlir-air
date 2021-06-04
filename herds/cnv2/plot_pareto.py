import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Plots paths')
parser.add_argument('-filename', default="", help='filename to be plotted')
parser.add_argument('-columnName', default="", help='Throughput or Latency')
args = parser.parse_args()

print(args.filename)

data = np.genfromtxt(args.filename, delimiter=' ', usecols=[0,1,2], names=True, dtype=int)

def plotTimes():
    plt.grid()
    plt.plot(np.arange(data['Area'][0], len(data['Area']) + data['Area'][0]), data[args.columnName], label=args.columnName)
    plt.xticks(np.arange(data['Area'][0], len(data['Area']) + data['Area'][0], 10))
    plt.legend()
    plt.ylabel("FPS")
    plt.show()

plotTimes()

