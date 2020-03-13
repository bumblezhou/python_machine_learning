import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def single_variate_logistic_regression():
    # Single-Variate Logistic Regression With scikit-learn
    # for given equation: y = 1 / (1 + exp(-x)),
    # this equation's shape is a sigmoid-curve which cross(0, 0.5) and its y has a limit of [0, 1]
    # create a dataset for input(x) and output(y)
    y = []
    x = []
    r = []
    for i in range(-10, 11):
        x.append(list([i]))
        value = 1 / (1 + np.exp(-i))
        result = 0
        if value > 0.5:
            result = 1
        y.append(value)
        r.append(result)

    print("x for equation: y = 1 / (1 + exp(-x)):")
    print(x)
    print("y for equation: y = 1 / (1 + exp(-x)):")
    print(y)
    print("r for equation: y = 1 / (1 + exp(-x)):")
    print(r)

    # Create a Model and Train It
    # You should carefully match the solver and regularization method for several reasons:
    # 'liblinear' solver doesn’t work without regularization.
    # 'newton-cg', 'sag', 'saga', and 'lbfgs' don’t support L1 regularization.
    # 'saga' is the only solver that supports elastic-net regularization.
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(x, r)
    # print classes, intercept, coefficient
    print("classes:", model.classes_)
    print('intercept:', model.intercept_)
    print('coefficient:', model.coef_)

    # predict the old data
    r_matrix = model.predict_proba(x)
    print('predicted matrix:', r_matrix)

    r_resp = model.predict(x)
    print('predicted response:', r_resp)

    # score for old data
    score = model.score(x, r)
    print('score:', score)

    # print classification report
    print(classification_report(r, model.predict(x)))

    # predict new data
    y_new = []
    x_new = []
    r_new = []
    for i in range(11, 20):
        x_new.append(list([i]))
        value = 1 / (1 + np.exp(-i))
        result = 0
        if value > 0.5:
            result = 1
        y_new.append(value)
        r_new.append(result)
    r_resp = model.predict(x_new)
    print('predicted response:', r_resp)

    y_new = []
    x_new = []
    r_new = []
    for i in range(-20, -10):
        x_new.append(list([i]))
        value = 1 / (1 + np.exp(-i))
        result = 0
        if value > 0.5:
            result = 1
        y_new.append(value)
        r_new.append(result)
    r_resp = model.predict(x_new)
    print('predicted response:', r_resp)

    # create the confusion matrix
    cm = confusion_matrix(r, model.predict(x))
    # show result on plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.show()


def multi_variate_logistic_regression():
    # Multi-Variate Logistic Regression With scikit-learn
    # The logit 𝑓(𝑥₁, 𝑥₂) = 𝑏₀ + 𝑏₁𝑥₁ + 𝑏₂𝑥₂
    # The probabilities 𝑝(𝑥₁, 𝑥₂) = 1 / (1 + exp(−𝑓(𝑥₁, 𝑥₂)))
    # create a dataset for inputs(x1, x2), outputs(y) and probabilities(r)
    y = []
    x = []
    r = []
    for i in range(-10, 11):
        # presume b0 = 0, b1 = 3, b2 = -2, it means that x1 has positive effect, and x2 has negative effect
        x1 = random.randint(-100, 100)
        x2 = random.randint(-100, 100)
        b0 = 0
        b1 = 3
        b2 = -2
        x.append(list([x1, x2]))
        value = 1 / (1 + np.exp(-(b0 + b1 * x1 + b2 * x2)))
        result = 0
        if value > 0.5:
            result = 1
        y.append(value)
        r.append(result)

    print("x for equation: y = 1 / (1 + exp(-x)):")
    print(x)
    print("y for equation: y = 1 / (1 + exp(-x)):")
    print(y)
    print("r for equation: y = 1 / (1 + exp(-x)):")
    print(r)

    # Create a Model and Train It
    # You should carefully match the solver and regularization method for several reasons:
    # 'liblinear' solver doesn’t work without regularization.
    # 'newton-cg', 'sag', 'saga', and 'lbfgs' don’t support L1 regularization.
    # 'saga' is the only solver that supports elastic-net regularization.
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(x, r)
    # print classes, intercept, coefficient
    print("classes:", model.classes_)
    print('intercept:', model.intercept_)
    print('coefficient:', model.coef_)

    # predict the old data
    r_matrix = model.predict_proba(x)
    print('predicted matrix:', r_matrix)

    r_resp = model.predict(x)
    print('predicted response:', r_resp)

    # score for old data
    score = model.score(x, r)
    print('score:', score)

    # print classification report
    print(classification_report(r, model.predict(x)))

    # predict new data
    y_new = []
    x_new = []
    r_new = []
    for i in range(-50, 51):
        # presume b0 = 0, b1 = 3, b2 = -2, it means that x1 has positive effect, and x2 has negative effect
        x1 = random.randint(-300, 300)
        x2 = random.randint(-300, 300)
        b0 = 0
        b1 = 3
        b2 = -2
        x_new.append(list([x1, x2]))
        value = 1 / (1 + np.exp(-(b0 + b1 * x1 + b2 * x2)))
        result = 0
        if value > 0.5:
            result = 1
        y_new.append(value)
        r_new.append(result)
    print("r_new for equation: y = 1 / (1 + exp(-x)):")
    print(r_new)
    r_resp = model.predict(x_new)
    print('predicted new response:', r_resp)

    # score for new data
    score = model.score(x_new, r_new)
    print('score:', score)

    # print classification report
    print(classification_report(r_new, model.predict(x_new)))


if __name__ == '__main__':
    # single_variate_logistic_regression()
    multi_variate_logistic_regression()
