# Capstone-Project
Bertelsmann/Arvato Project

### libraries used 
-numpy
import numpy as np

-pandas
import pandas as pd

-matplotlib
import matplotlib.pyplot as plt

-seaborn
import seaborn as sns

-math
import math

-pickle
import pickle

-sklearn
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

-tarfile
import tarfile

-urllib
import urllib

### motivation for the project
In this project, demographics data for customers of a mail-order sales company in Germany will be analysed, comparing it against demographics information for the general population. Unsupervised learning techniques will be used to perform customer segmentation, identifying the parts of the population that best describe the core customer base of the company. Then, a third dataset with demographics information for targets of a marketing campaign for the company will be used, and a model to predict which individuals are most likely to convert into becoming customers for the company. The used has been provided by Udacity’s partners at Bertelsmann Arvato Analytics and represents a real-life data science task.
The data sets used in this project are:
•	Udacity_AZDIAS_052018.csv: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
•	Udacity_CUSTOMERS_052018.csv: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
•	Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
•	Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns)
The problem which needs to be solved is a customer segmentation which classifies customers into groups of those likely to be converted into becoming customers and those who are not likely. This will help the company decide on which customers to target in their marketing campaign.
To do this I plan to:
1)	Assess the customer and azdias datasets to identify what cleaning needs to be done to the datasets to prepare them for analysis.
2)	Clean the datasets in the way identified in step 1.
3)	Analyse the datasets using PCA and k-means clustering to identify the parts of the population that best describe the core customer base of the company.
4)	Pre-process the training dataset by cleaning it and standardising the values in it.
5)	Use classifiers to model the data.

6)	Test the classifiers and compare MAE and accuracy scores to decide the best model.
7)	Find best model parameters through tuning.
8)	Apply best model to test data and generate predictions.
9)	Find the set of customers that are likely to become customers.

I will be using the accuracy as the metrics to measure my model’s performance as it measures how often the classifier correctly predicts, which is an important factor of a classification model.

My motivation for this project is because I work in the financial sector and this is the kind of work I am hoping to complete in my job.

### files in the repository
README- Information about the project 

Arvato Project Workbook.ipyn- Code for project 

Capstone Project Report.docx- Word document of Report which is published as a blog here: https://medium.com/@sarahdalton1988/capstone-project-8cfdb0af0d49

DIAS Attributes- Values 2017 (1).xlsx- Excel file containing the different attributes used in the datasets with a description and the codes

DIAS Information Levels- Attributes 2017(1).xlsx- Excel file containing the inforamtion levels of the different attributes from the datasets 

### summary of the results of the analysis
When I applied the model and tried to find the set of customers who were likely to become customers, the results came back as zero customers, which is not what I would have liked to see!

### necessary acknowledgements
Stack Overflow
Kaggle
Medium
https://builtin.com/data-science/supervised-machine-learning-classification

The blog post which contains the report on this project can be found here: https://medium.com/@sarahdalton1988/capstone-project-8cfdb0af0d49
