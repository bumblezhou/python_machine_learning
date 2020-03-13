import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def simple_linear_regression():
    # Simple Linear Regression With scikit-learn
    # for given equation: y = 5.5 + 4.7 * x,
    # create a dataset for input(x) and output(y)
    y = []
    x = []
    for i in range(-100, 100):
        x.append(list([i]))
        y.append(5.5 + 4.7 * i)

    print("x for equation: y = 5.5 + 4.7 * x:")
    print(x)
    print("y for equation: y = 5.5 + 4.7 * x:")
    print(y)

    # Create a model and fit it
    model = LinearRegression().fit(x, y)
    # Get results
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)

    # Predict response
    y_pred = model.predict(x)
    print('predicted response:', y_pred, sep='\n')

    # Predict new data
    x_new = [list([item]) for item in range(101, 151)]
    print('x_new:', x_new, sep='\n')
    y_new = model.predict(x_new)
    print('predicted new:', y_new, sep='\n')


def multiple_linear_regression():
    # Multiple Linear Regression With scikit-learn
    # for given equation: y = 3.6 + 2.5 * a + 4.7 * m - 5.4 * n,
    # create a dataset for input(a, m, n) and output(y)
    y = []
    x = []
    for i in range(-100, 100):
        a = random.randint(0, 100)
        m = random.randint(0, 100)
        n = random.randint(0, 100)
        x.append(list([a, m, n]))
        y.append(3.6 + 2.5 * a + 4.7 * m - 5.4 * n)

    print("[a, m, n] for equation: y = 3.6 + 2.5 * a + 4.7 * m - 5.4 * n:")
    print(x)
    print("y for equation: y = 3.6 + 2.5 * a + 4.7 * m - 5.4 * n:")
    print(y)

    # Create a model and fit it
    model = LinearRegression().fit(x, y)
    # Get results
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)

    # Predict response
    y_pred = model.predict(x)
    print('predicted response:', y_pred, sep='\n')

    # Predict new data
    x_new = [list([item, item, item]) for item in range(101, 151)]
    print('x_new:', x_new, sep='\n')
    y_new = model.predict(x_new)
    print('predicted new:', y_new, sep='\n')


def ploynomial_regression():
    # Polynomial Regression With scikit-learn
    # Combinations of polynomial:
    # [a] combinations of a square polynomial: [1, a, a*a]
    # [a, b] combinations of a square polynomial: [1, a, b, a*a, ab, b*b]

    # for given equation: y = 3 - (5 * a) + (6 * b) + 4 * (a * a) - 7 * a * b + 2 * (b * b),
    # create a dataset for input(a, b) and output(y)
    y = []
    x = []
    for i in range(0, 6):
        random.seed(i ^ 3 - 35)
        a = random.randint(-100, 100)
        random.seed(i ^ 2 + 67)
        b = random.randint(-100, 100)
        x.append(list([a, b]))
        value = 3 - (5 * a) + (6 * b) + 4 * (a * a) - 7 * a * b + 2 * (b * b)
        y.append(value)
    print("[a] for equation: y = 3 - (5 * a) + (6 * b) + 4 * (a * a) - 7 * a * b + 2 * (b * b):")
    print(x)
    print("y for equation: y = 3 - (5 * a) + (6 * b) + 4 * (a * a) - 7 * a * b + 2 * (b * b):")
    print(y)

    # Transform input data
    x_ = PolynomialFeatures(degree=2).fit_transform(x)
    print("transform result for x:")
    print(x_)

    # Create a model and fit it
    model = LinearRegression(fit_intercept=False).fit(x_, y)
    # Get results
    r_sq = model.score(x_, y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('equation: y = 3 - (5 * a) + (6 * b) + 4 * (a * a) - 7 * a * b + 2 * (b * b)')
    print("expected slope:[3, -5, 6, 4, -7, 2]")
    print('slope:')
    print(', '.join('{:.2f}'.format(f) for f in model.coef_))

    # Predict response
    print('predicted [[28, 47]]:')
    print('expected:{}'.format(3 - (5 * 28) + (6 * 47) + 4 * (28 * 28) - 7 * 28 * 47 + 2 * (47 * 47)))
    x = [[28, 47]]
    x_ = PolynomialFeatures(degree=2).fit_transform(x)
    y_pred = model.predict(x_)
    print('predicted result:', y_pred, sep='\n')


if __name__ == '__main__':
    # simple_linear_regression()
    # multiple_linear_regression()
    ploynomial_regression()
