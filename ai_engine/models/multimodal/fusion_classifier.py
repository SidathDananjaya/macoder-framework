from sklearn.ensemble import RandomForestClassifier


class FusionClassifier:

    def __init__(self):

        self.model = RandomForestClassifier(

            n_estimators=200,
            max_depth=10,
            random_state=42
        )

    def train(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        return self.model.predict(X)