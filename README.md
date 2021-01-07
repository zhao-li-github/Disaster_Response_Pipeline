# Disaster Response Pipeline Project

### Content:
1. "Data" folder:
-- "process_data.py": this is the Python codes to read in the data (messages and categories), cleans and save it in a SQL database
-- "disaster_categories.csv": the CSV file that contains the categories to be classified into
-- "disaster_messages.csv": the CSV file that contains the disaster response messages
-- "cleaned_data": the SQL database created by "process_data.py" that contains the merged and cleaned data

2. "Models" folder:
-- "train_classifier.py": this is the Python codes to load the cleaned SQL database, tokenize the text, then build a machine learning (ML) pipeline based on RandomForestClassifier and train the model, then save the model in a pickle file (the pickle file is too big and therefore not uploaded here).

3. "App" folder:
-- "run.py": Flask app and the user interface to classify a user-entered text based on the trained model and display the result
-- "templates" folder: the folder that conntains the html templates

4. "ETL Pipeline Preparation.ipynb" and "ML Pipeline Preparation.ipynb" are two Jupyter notebooks that help me go through the pipeline buildign process. You can also find some data visualization figures there. "DisasterResponseProject_test-run.png" is a screenshot for a test run on the interface.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/cleaned_data`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/cleaned_data model/trained_model`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
