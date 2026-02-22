def evaluate(model, X_test, y_test):

    correct = 0

    for i in range(len(X_test)):

        x1 = X_test[i][0]
        x2 = X_test[i][1]
        y_true = y_test[i]

        z = model.linear(x1, x2)
        y_pred = model.activation(z)

        final_pred = 1 if y_pred > 0.5 else 0

        if final_pred == y_true:
            correct += 1

    accuracy = correct / len(y_test)
    return accuracy
