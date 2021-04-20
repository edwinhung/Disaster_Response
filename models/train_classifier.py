import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import joblib

def load_data(database_filepath):
    """
    Load data from database and pull training set and target

    Args:
        database_filepath (str): filepath of database

    Returns:
        series: a columne of message text as training data
        dataframe: multioutput targe columns
        index: categories labels
    """
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql("SELECT * FROM Response",engine)
    X = df["message"]
    y = df.iloc[:,4:]
    return X, y, y.columns


def tokenize(text):
    """
    Function to tokenize text using NLP pipeline with lemmatization

    Args:
        text (str): original text

    Returns:
        list of str: tokens of text
    """
    text = re.sub("[^a-zA-Z0-9]"," ",text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    return clean_tokens


def build_model():
    """
    Function to build ML pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LGBMClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to print classification report of predictions on text set

    Args:
        model: model to be evaluated
        X_test (series): test set of message
        Y_test (dataframe): test set of categories
        category_names (index): categories labels

    Print:
        classification result
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test,Y_pred,target_names=Y_test.columns))



def save_model(model, model_filepath):
    """
    Save model in pickle file
    """
    filename = model_filepath
    joblib.dump(model, filename)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()