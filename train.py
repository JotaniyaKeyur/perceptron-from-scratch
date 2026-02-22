def train(model, X_train, y_train, epochs):

    for epoch in range(epochs):

        total_loss = 0

        for i in range(len(X_train)):

            x1 = X_train[i][0]
            x2 = X_train[i][1]
            y_true = y_train[i]

            z = model.linear(x1, x2)
            y_pred = model.activation(z)

            L = model.loss(y_pred, y_true)

            model.backward(x1, x2, y_true, y_pred)

            total_loss += L

        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(X_train)}")

    return model
