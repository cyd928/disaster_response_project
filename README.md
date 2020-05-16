# Disaster Response Pipeline Project

## Project Overview
* The aim of this project is to build a machine learning model pipeline to forecast the categories of disaster response messages. The result of the model prediction is illustrated by a web app 

	(Link to the web app: http://0.0.0.0:3001/).


## Web App 
This web app serves two main functions:
* Desmonstrate the characteristics of the training dataset
* Allow users to input messages for the model to predict message categories

## File Explanations
* ```app```
	* ```templates```: html templates for the web design
	* ```run.py```: python script to get the contents (visualisations, model) to the app

* ```data```
	* ```.csv```files: raw datasets of disaster responses
	* ```process_data.py```: python script for ETL process
	* ```DisasterResponse.db```: database where the post-processed data is stored

* ```model```
	* ```train_classifier.py```: python script for building machine learning model using db data

* ```Procfile```: code to call gunicorn and run web app

* ```README.md```: cover page / project description fie

* ```Requirements.txt```: packages required to run the scripts


