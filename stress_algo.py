import sqlite3
import json
from flask import Flask, request, jsonify, session, g, redirect, url_for, abort, render_template, flash
from contextlib import closing
import validation

# Import Pickle
import pickle

# ML Imports
import pandas as pd
from pandas import Series,DataFrame

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

import seaborn as sns

# configuration
DATABASE = 'affektive.db'
DEBUG = True
SECRET_KEY = 'development key'
USERNAME = 'admin'
PASSWORD = 'default'

#create app
app = Flask(__name__)
app.config.from_object(__name__)
app.config.from_envvar('APP_SETTINGS', silent=True)

def connect_db():
	return sqlite3.connect(app.config['DATABASE'])

def init_db():
	with closing(connect_db()) as db:
		with app.open_resource('schema.sql', mode='r') as f:
			db.cursor().executescript(f.read())
		db.commit()

def start_db_request():
	g.db = connect_db()

def end_db_request():
	db = getattr(g, 'db', None)
	if db is not None:
		db.close()

def process_data(userid):
	with app.app_context():
		start_db_request()
		cur = g.db.execute('select userid, timestamp, hr, gsr, state, level from entries where userid=?', (userid,))

		# Preprocessing Data
                df = pd.DataFrame(cur.fetchall())
		df.columns = ['userid','time','hr','gsr','state','level']
                df.to_pickle('training_set.pkl') # Save local copy of DataFrame

                # Data Visualizations
		normal = df[df['state'] == 'Normal']
		calm = df[df['state'] == 'Calm']
		stressed = df[df['state'] == 'Stressed']

		ax = normal.plot(kind='scatter', x='hr', y='gsr',
						color='DarkBlue', label='Normal')
		calm.plot(kind='scatter', x='hr', y='gsr',
						color='DarkGreen', label='Calm', ax=ax)
		stressed.plot(kind='scatter', x='hr', y='gsr',
						color='DarkRed', label='Stressed', ax=ax)

		plt.savefig('output.png')

                # Convert to Numpy Format
                le = preprocessing.LabelEncoder()
                le.fit(df['state'])
                df['state'] = le.transform(df['state'])
                X = df[['hr','gsr']].values
                T = df[['state']].values

                # Creating our Training and Validation Set using hold-out
                X_train, X_valid, t_train, t_valid = train_test_split(X, T, test_size=0.25, random_state=42)

                # Train our Model using the Training Set
                neigh = KNeighborsClassifier(n_neighbors=4)
                neigh.fit(X_train,t_train.T[0])

                # Forming a Validation Set to Evaluate our model
                y_valid = neigh.predict(X_valid)
                valid_num_correct  = float(np.sum(np.equal(y_valid,t_valid.T[0])))
                print valid_num_correct
                print t_valid.shape[0]
                classification_rate = float(valid_num_correct/t_valid.shape[0])*100.0
                print classification_rate

                # Classify unseen data into one of the states and quantify level



                end_db_request()
		return

if __name__ == "__main__":
	init_db()
	process_data("affektive")

