# Disaster Response Machine Learning Pipeline

## Summary
The repository contains code for a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.  The repository contains a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

## Running Instructions
### ***Run ETL Process***
1. To run ETL pipeline that cleans data and stores in database:<br>
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

### ***Run ML Pipeline***
1. To run ML pipeline that trains classifier and saves:<br>
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

### ***Run Web Application***
**Running the web app locally**
1. Run the following command in the app directory:<br>
    `python run.py`
2. In a browser navigate to http://localhost:3001/

**Run the app from a Udacity VM**
1. You need to find the workspace environmental variables with `env | grep WORK`, and you can open a new browser window and go to the address: <br>
    `http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN` <br>
replacing WORKSPACEID and WORKSPACEDOMAIN with your values.

## Files
* **ETL Pipeline Preparation.ipynb**: Jupyter notebook of code and analysis used in the development of process_data.py.
* **ML Pipeline Preparation.ipynb**: Jupyter notebook of code and analysis used in the development of train_classifier.py.
* **process_data.py**: Takes disaster_categories.csv and disaster_messages.csv files (or other files specified as command line parameters), and it creates an SQLite database named DisasterResponse.db (or other DB specified as command line parameter) with the merged and cleaned data.
* **train_classifier.py**: Takes the SQLite database created by process_data.py (DisasterResponse.db or another specified from the command line).  The data is used to train a machine learning model.  The code outputs a pickle file of the fitted model.
* **disaster_categories.csv**: Default disaster category data used by the process_data.py file.
* **disaster_messages.csv**: Default disaster messages data used by the process_data.py file.
* **DisasterResponse.db**: Default SQLLite database data used by process_data.py file and train_classifier.py
* **classifier.pkl**: Default pickle file used by the train_classifier.py file.
* **run.py**: Code that runs the webb app


## Note
The data sourced for this program was provided by [Udacity](https://www.udacity.com) and [Figure Eight](https://www.figure-eight.com/).  The dataset is not very evenly distributed between postive and negative examples.  This can skew the results of the classifier and provide misleading accuracy.
