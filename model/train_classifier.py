import sys

##Data extraction packages
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

## Data transform packages
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

## Machine learning packages
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Model improvement
from sklearn.model_selection import GridSearchCV

# Save model
import pickle


def load_data(database_filepath):
    '''load data from sql database
    Args:
        X: list of messages
        Y: dataframe of responses
        category_names: categories that responses belong to
    '''
    engine = create_engine('sqlite:///{}'.format(str(database_filepath)))
    df = pd.read_sql_table('msgcat', engine)
    X = df.message
    Y = df.iloc[:,4:]
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    '''Tokensize text data'''
    tokens = word_tokenize(text.lower().strip())
  
    clean_tokens = []
    for tok in tokens:
        clean_tok = WordNetLemmatizer().lemmatize(tok)
        clean_tokens.append(clean_tok)
    return clean_tokens



def build_model():
    '''build multi output classifier of random forest model on the tokenized data;
    use grid search to improve model accuracy
    '''
    model = Pipeline([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
            ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier())
        )
    ])
    
    parameters = {
        'clf__estimator__criterion': ['entropy'],
        'clf__estimator__max_depth': [5, None],
        'clf__estimator__n_estimators': [20,50]
        }

    cv = GridSearchCV(model, param_grid=parameters)
    
    return cv




def evaluate_model(model, X_test, Y_test, category_names):
    ''' use test data to evaluate model accuracy;
    create classification report on the result of each category
    
    '''
    Y_pred = model.predict(X_test)

    for i in range(0, len(category_names)):
        print(category_names[i],': \n',
          classification_report(Y_test.iloc[:,i], Y_pred[:,i]))
        
    pass


def save_model(model, model_filepath):
    Pkl_File = model_filepath

    with open(Pkl_File, 'wb') as file:  
        pickle.dump(model, file)
        
    pass


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