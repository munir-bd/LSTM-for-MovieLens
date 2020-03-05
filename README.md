# LSTM-for-MovieLens
Project focus on LSTM model for MovieLens. 
Data preproced by TA (Mr. Kyi Thar) of the course https://github.com/kyithar/class
We analyze EDA and implement our LSTM model for preproced MovieLens dataset.

Predicting Popularity Scores by utilizing Recurrent Neural Network
(RNN)
Project Description
In this project, you may need to predict the popularity score of movies, where the main objective is to
improve the prediction accuracy as much as you can. In order to achieve this goal, you can implement the
prediction model by using Recurrent Neural Network (RNN) model (such as Long Short Term Memory
(LSTM), Gated Recurrent Unit (GRU)), and also you can use the hybrid model (the combination of
Convolutional Neural Network (CNN) + RNN).
Requirements
Dataset: MovieLens (http://files.grouplens.org/datasets/movielens/ml-latest.zip)
Package need to install: pandas, numpy,matplotlib, seaborn
Python version: 3.6
Machine learning libraries: tensorflow, keras, pytorch
Dataset property
This dataset (ml-latest) describes 5-star rating and free-text tagging activity from (http://movielens.org),
a movie recommendation service. It contains 26024289 ratings and 753170 tag applications across 45843
movies. These data were created by 270896 users between January 09, 1995 and August 04, 2017. This
dataset was generated on August 04, 2017.
Original dataset formatting and encoding (please check README.txt)
*Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.*
Data cleaning process
You can download sample code from the Github link preprocess folder (https://github.com/kyithar/class)
It includes
1. Join_dataset.py (To join two csv files)
2. Preprocessing.py (To calculate the count and normalized count to make label)
3. Sort.py (Sort dataset with timestamp (second))
4. to_seq.py (To get sequences of inputs)

