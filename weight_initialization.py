"""weight_initialization 
~~~~~~~~~~~~~~~~~~~~~~~~

This program shows how weight initialization affects training.  In
particular, we'll plot out how the classification accuracies improve
using either large starting weights, whose standard deviation is 1, or
the default starting weights, whose standard deviation is 1 over the
square root of the number of input neurons.

"""

# Standard library
import sys 
import json
import random
import mnist_loader
import network2

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

enpoch = 30


def main():
    filename = sys.argv[1]
    n = int(sys.argv[2])
    eta = float(sys.argv[3])
    run_network(filename, n, eta)
    make_plot(filename)


def run_network(filename, n, eta):
    """Train the network using both the default and the large starting
    weights.  Store the results in the file with name ``filename``,
    where they can later be used by ``make_plots``.

    """
    # Make results more easily reproducible
    random.seed(12345678)
    np.random.seed(12345678)

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net1 = network2.Network([784, n, 10], cost=network2.CrossEntropyCost)
    print("Train the network using the large starting weights.")
    net1.large_weight_initializer()
    large_vc, large_va, large_tc, large_ta = net1.SGD(training_data, enpoch, 10, eta, lmbda=5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net2 = network2.Network([784, n, 10], cost=network2.CrossEntropyCost)
    print("Train the network using the default starting weights.")
    default_vc, default_va, default_tc, default_ta = net2.SGD(training_data, enpoch, 10, eta, lmbda=5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)

    f = open(filename, "w")
    json.dump({"default_weight_initialization": [default_vc, default_va, default_tc, default_ta],
               "large_weight_initialization": [large_vc, large_va, large_tc, large_ta]},
              f)
    f.close()


def make_plot(filename):
    """Load the results from the file ``filename``, and generate the
    corresponding plot.

    """
    f = open(filename, "r")
    results = json.load(f)
    f.close()
    default_vc, default_va, default_tc, default_ta = results["default_weight_initialization"]
    large_vc, large_va, large_tc, large_ta = results["large_weight_initialization"]
    # Convert raw classification numbers to percentages, for plotting
    default_va = [x / 100.0 for x in default_va]
    large_va = [x / 100.0 for x in large_va]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, enpoch, 1), large_va, color='#2A6EA6', label="Old approach to weight initialization")
    ax.plot(np.arange(0, enpoch, 1), default_va, color='#FFA933', label="New approach to weight initialization")
    ax.set_xlim([0, enpoch])
    ax.set_xlabel('Epoch')
    ax.set_ylim([85, 100])
    ax.set_title('Classification accuracy')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()
