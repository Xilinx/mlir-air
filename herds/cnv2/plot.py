import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Plots paths')
parser.add_argument('-filename', default="", help='filename to be plotted')
args = parser.parse_args()

print(args.filename)

labels = np.genfromtxt(args.filename, delimiter=" ", usecols=0, dtype=str, skip_header=1)
data = np.genfromtxt(args.filename, delimiter=' ', usecols=[1,2,3,4,5,6,7,8,9,10], names=True, dtype=int)

def plotTimes():
    plt.plot(data['Compute'], label='ComputeTime')
    plt.plot(data['ActCommunication'], label='ActCom')
    plt.plot(data['WeightCommunication'], label='WeightCom')
    plt.xticks(np.arange(len(labels)), labels, rotation='vertical')
    plt.legend()
    plt.ylabel("cycles")
    plt.show

plotTimes()

