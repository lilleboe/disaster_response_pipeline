import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMsgTable', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals  
    genre_counts = df.groupby('genre').count()['message']
    genre_names  = list(genre_counts.index)
    
    # extract the sums of each category type
    category = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category = category.sum()
    category = category.sort_values(ascending = False)
    cat_names = list(category.index)
    
    # extract the sums of each category type
    news = df[df['genre'] == 'news']
    news = news.drop(['id', 'message', 'original', 'genre'], axis = 1)
    news = news.sum()
    news = news.sort_values(ascending = False)
    news_names = list(news.index)
    
    direct = df[df['genre'] == 'direct']
    direct = direct.drop(['id', 'message', 'original', 'genre'], axis = 1)
    direct = direct.sum()
    direct = direct.sort_values(ascending = False)
    direct_names = list(direct.index)
  
    social = df[df['genre'] == 'social']
    social = social.drop(['id', 'message', 'original', 'genre'], axis = 1)
    social = social.sum()
    social = social.sort_values(ascending = False)
    social_names = list(social.index)
  
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
		},
		{	
            'data': [
                Bar(
                    x=cat_names,
                    y=category
                )
            ],

            'layout': {
                'title': 'Messages by Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -45
                }
            }
        },
		{	
            'data': [
                Bar(
                    x=news_names,
                    y=news,
                    name='news'
                ),
                Bar(
                    x=direct_names,
                    y=direct,
                    name='direct'
                ),
                Bar(
                    x=social_names,
                    y=social,
                    name='social'
                )
            ],

            'layout': {
                'title': 'Messages by Genre and Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -45
                }
            }
        }
    ]
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()