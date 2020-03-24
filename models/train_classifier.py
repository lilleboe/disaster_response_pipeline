import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sqlalchemy import create_engine

import warnings
warnings.simplefilter('ignore')

def load_data(database_filepath):
    """Loads data from a SQLLite database
    
    Args:
    database_filepath: String. Filepath of the SQLLite database
    
    Returns:
    X: dataframe. Contains features dataset
    y: dataframe. Contains labels dataset
    column names: list of strings. List containing categories names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterMsgTable', engine)  
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    return X, y, list(y.columns.values)


def tokenize(text):
    """Method to normalize-->tokenize-->lemmatize-->Stem
    
    Args:
    text: String. Message data
       
    Returns:
    cleaned_data: A list of cleaned strings
    """
    
    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize the text data
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tok = re.sub(r"[^a-zA-Z0-9]", " ", clean_tok.lower())
        clean_tok = clean_tok.strip()
        
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """Builds a machine learning model via a pipeline
    
    Args:
    None
       
    Returns:
    gs: GridSearchCV object. Transforms data and builds 
    model via optimal model parameters.
    """
    # Create the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'vect__max_df': (0.5, 1.0),
                  'vect__ngram_range': ((1, 1), (1, 2)),  #unigrams or bigrams
                  'tfidf__use_idf':[True, False],
                  'tfidf__norm': ('l1', 'l2'),
                  'clf__estimator__n_estimators':[50, 100], 
                  'clf__estimator__min_samples_split':[2, 5]}

    gs = GridSearchCV(pipeline, param_grid=parameters)

    return gs


def evaluate_model(model, X_test, Y_test, category_names):
    """Calculate and disply precision, recall, f1-score, accuracy
    
    Args:
    actual: array. actual label values.
    predicted: array. predicted label values.
    columns: list. List of field names
       
    Returns: None
    """
    
    # Predict labels for test dataset
    y_pred = model.predict(X_test)
    
    classify = classification_report(Y_test, y_pred, target_names=category_names)    
    print('Classification Report\n', '-' * 75)
    print(classify, '\n')
    
    # Display accuracy
    accuracy = (Y_test == y_pred).mean()
    print('Accuracy\n', '-' * 75)
    print(accuracy)
    print('\nAverage Accuracy:', accuracy.mean())


def save_model(model, model_filepath):
    """Save pickle model
    
    Args:
    model: model object. Fitted model object
    model_filepath: string. Filepath for saved model
    
    Returns:
    None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


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