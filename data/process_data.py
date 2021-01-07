# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load messages and categories

    Parameters: filepath for both messages.csv and categories.read_csv
    Return: a merged pd dataframe of both messages.csv and categories.read_csv
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge the two dataframes
    df = messages.merge(categories, on='id')

    return df

def clean_data(df):
    '''
    Clean the dataframe by separating the category column into individual columns,
    convert category values into numbers, and drop duplicated rows

    Parameters: filepath for both messages.csv and categories.read_csv
    Return: a merged pd dataframe of both messages.csv and categories.read_csv
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';',expand=True)
    # find the categories' names using this new dataframe
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of the new "categories" dataframe
    categories.columns = category_colnames

    # convert category values to just numbers: 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # When I checked the new categories dataframe, I noticed that in the "related" column,
    # there are unexpected values of "2". After comparing to value 0 and 1 in the "related"
    # column on the distribution of other 35 features, I concluded that the unexpected
    # value of "2" is most likely due to a typo and should be correct into "0". More
    # details and figures can be found in the "ETL Pipeline" Jupyter notebook
    categories.loc[categories['related'] == 2,'related'] = 0

    # replace categories column in the df dataframe with new categories category_colnames
    # drop the original categories column from the df dataframe
    df.drop(columns='categories', axis=1, inplace=True)
    # concat the original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    '''
    Save Data to SQLite Database

    Parameters: dataframe to be saved, and the filename for the SQLite database
    Return: None
    '''

    engine = create_engine('sqlite:///'+database_filename+'.db')
    df.to_sql('Cleaned_Data', engine, if_exists = 'replace', index=False)
    engine.dispose()


# This part is provided by the course, no change is made
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
