import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)
# tokenize function was not included in pipeline, so we need to define function
# here to load model from pickle file
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Response', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Bar chart of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    subjects = ['aid_related','infrastructure_related','weather_related']
    subjects_counts = df[subjects].sum()
    subjects_names = list(subjects_counts.index)

    sub_graphs = [
        {
            'data': [
                Bar(
                    x=subjects_names,
                    y=subjects_counts,
                    marker={'color':'orange'}
                )
            ],

            'layout': {
                'title': 'Bar chart of Subjects',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Subject"
                }
            }
        }
    ]

    sub_ids = ["graph-{}".format(i+10) for i, _ in enumerate(sub_graphs)]
    sub_graphJSON = json.dumps(sub_graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, 
                            sub_ids=sub_ids, sub_graphJSON=sub_graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()