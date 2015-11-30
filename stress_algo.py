import sqlite3
import json
from flask import Flask, request, jsonify, session, g, redirect, url_for, abort, render_template, flash
from contextlib import closing
import validation

# ML Imports
import pandas as pd
from pandas import Series,DataFrame

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.svm import SVC

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
		#print cur.fetchall()[0]
		#print type(cur.fetchall()[0])
		#data = np.asarray(cur.fetchall())
		df = pd.DataFrame(cur.fetchall())
		df.columns = ['userid','time','hr','gsr','state','level']
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

		end_db_request()
		return

if __name__ == "__main__":
	init_db()
	process_data("affektive")

