from data_loader import load_data
from model import PerceptronScratch
from train import train
from evaluate import evaluate


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_data("placement-dataset.csv")

    model = PerceptronScratch(w1=0.1, w2=0.2, b=-0.1, lr=0.05)

    model = train(model, X_train, y_train, epochs=200)

    accuracy = evaluate(model, X_test, y_test)

    print("\nFinal Weights:")
    print("w1:", model.w1)
    print("w2:", model.w2)
    print("b :", model.b)

    print("\nAccuracy:", accuracy)
