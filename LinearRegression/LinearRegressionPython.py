import matplotlib.pyplot as plt
import numpy as np

x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T
y = np.array([[2,4 ,3, 6, 9, 12, 13, 15, 18, 20]]).T


def calculateB0B1(x, y):
    # Calculate the average
    xbar = np.mean(x)
    ybar = np.mean(y)
    x2bar = np.mean(x ** 2)
    x_ybar = np.mean(x * y)

    # calculate b0, b1
    b1 = (xbar * ybar - x_ybar) / (xbar ** 2 - x2bar)
    b0 = ybar - b1 * xbar
    return b1, b0


b1, b0 = calculateB0B1(x, y)
print("b1:", b1)
print("b0:", b0)

y_predicted = b0 + b1 * x
print(y_predicted)


# Visualize data
def showGraph(x, y, y_predicted, title="", xlabel="", ylabel=""):
    plt.figure(figsize=(14, 8))
    plt.plot(x, y, 'r-o', label="value sample")
    plt.plot(x, y_predicted, 'b-x', label="predicted value")

    x_min = np.min(x)
    y_min = np.min(y)
    x_max = np.max(x)
    y_max = np.max(y)

    # Mean y value
    ybar = np.mean(y)

    plt.axhline(ybar, linestyle='--', linewidth=4, label="mean")
    plt.axis([x_min * 0.95, x_max * 1.05, y_min * 0.95, y_max * 1.05])
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.text(x_min + 1.03, ybar, "mean", fontsize=16)
    plt.legend(fontsize=15)
    plt.title(title, fontsize=20)
    plt.show()


showGraph(x, y, y_predicted,
          title="Y values corresponding to X",
          xlabel="X values",
          ylabel="Y values")
