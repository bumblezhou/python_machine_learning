import mnist_loader
import network


def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # the 28 * 28 = 784 pixels in a single MNIST image.
    # 30 hidden layers
    # result lays 10
    net = network.Network([784, 50, 10])
    # 30 epoch, mini-batch 10, η(learning rate) = 3.0
    net.SGD(training_data, 20, 10, 3.0, test_data=test_data)


if __name__ == '__main__':
    main()