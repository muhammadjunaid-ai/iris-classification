import joblib
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_and_preprocess

def evaluate():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

    model = joblib.load('model.pkl')

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate()