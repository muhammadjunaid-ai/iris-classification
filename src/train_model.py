from sklearn.linear_model import LogisticRegression
import joblib
from data_preprocessing import load_and_preprocess

def train():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

    model = LogisticRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    print("Model trained and saved!")

if __name__ == "__main__":
    train()