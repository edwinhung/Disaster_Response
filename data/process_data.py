import sys
import pandas as pd

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sqlalchemy import create_engine

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
    stopwords_list = stopwords.words("english")
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        if (clean_token not in stopwords_list): clean_tokens.append(clean_token)
    return clean_tokens

def load_data(messages_filepath, categories_filepath):
    """
    Function to load datasets from filepaths

    Returns:
        dataframe: merged dataframe from two datasets
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on="id")
    return df
    
def clean_data(df):
    """
    Transform categories labels to columns and clean data errors

    Args:
        df (dataframe): merged dataframe containing message and categories

    Returns:
        df (dataframe): clean dataframe
    """
    categories = df["categories"].str.split(";",expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = pd.to_numeric(categories[column])
    categories.replace(2,1,inplace=True)
    df.drop("categories",axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)
    df = df[~df.duplicated()]
    return df


def save_data(df, database_filename):
    """
    save dataframe in database

    Retruns:
        None
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Response', engine, index=False)

def save_words_data(df, database_filename):
    """
    save words series in database

    Retruns:
        None
    """
    message = df["message"]
    words = pd.concat(pd.Series(tokenize(t), name="word") for t in message)
    engine = create_engine('sqlite:///'+database_filename)
    words.to_sql('Words', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Saving words data...\n    DATABASE: {}'.format(database_filepath))
        save_words_data(df, database_filepath)

        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()