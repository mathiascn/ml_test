import pandas as pd
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import joblib

def main():
    
    def _train_model(data: DataFrame, filename: str) -> float:
        X = data.drop(columns=['genre'])
        y = data['genre']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = DecisionTreeClassifier()
        model.fit(X_train.values, y_train.values)
        
        joblib.dump(model, f'models/{filename}.joblib')
        tree.export_graphviz(model,
                            out_file=f'visualizations/{filename}.dot',
                            feature_names=['age', 'gender'],
                            class_names=sorted(y.unique()),
                            label='all',
                            rounded=True,
                            filled=True)
        
        predictions = model.predict(X_test.values)
        
        return accuracy_score(y_test, predictions)
        
    def _predict(input_data, filename):
        model: DecisionTreeClassifier = joblib.load(f'models/{filename}.joblib')
        predictions = model.predict([input_data])
        return predictions
    
    filename = 'music-recommender'
    music_data = pd.read_csv('datasets/music.csv')
    accuracy = _train_model(music_data, filename)
    print(f"Training accuracy: {accuracy:.2f}")

    predicted_genre = _predict([22,1], filename)
    print(f"Predicted genre for input [22,1]: {predicted_genre}")

if __name__ == '__main__':
    main()
