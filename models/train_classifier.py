# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
import pickle

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def load_data(database_filepath):
    '''
    Load the cleaned data

    Parameters: filepath for the database
    Return:
    X (pd.Series): message text
    Y (pd.DataFrame): dataframe that contains the dummy variables of each category
    category_nname (list): list that contains the categories' names
    '''

    engine = create_engine('sqlite:///'+database_filepath+'.db')
    df = pd.read_sql("SELECT * FROM Cleaned_Data", engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    return X, Y, category_names

def tokenize(text):
    '''
    Clean and tokeize the input text

    Parameters: input text
    Return:
    clean_token: tokens of the input text
    '''

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model(grid_search_cv=False):
    '''
    Build the ML model

    Parameters: whether to do a grid search (controled by "grid_search_cv", default is False)
    Return:
    pipeline: ML model
    '''

    forest = RandomForestClassifier(verbose=5)
    clf = MultiOutputClassifier(forest)
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', clf)
                    ])
    if grid_search_cv == True:
        print('Searching for best parameters...')
        parameters = {'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
        'vect__max_df': (0.75, 1.0),
        'vect__max_features: (5000, 10000)'
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]}

        pipeline = GridSearchCV(pipeline, param_grid=parameters, verbose=5)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Use the input model to predict on the test set, and then evaluate the model performance

    Parameters:
    model: trained ML model
    X_test: X dataset of the test set
    Y_test: Y_true of the test set
    category_names: categories' names
    Return:
    None; will print the evaluation results
    '''

    Y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, Y_pred, target_names=Y_test.columns.values))


def save_model(model, model_filepath):
    '''
    Save and export the trained model as a pickle file

    Parameters:
    model: trained ML model
    model_filepath: destination pickle filename
    Return: None
    '''
    pickle.dump(model, open(model_filepath+'.pkl', 'wb'))

# This part is provided by the course, no change is made
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
