import sqlite3
import json
from flask import Flask, request, jsonify, session, g, redirect, url_for, abort, render_template, flash
from contextlib import closing
import validation

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

@app.route('/')
def main():
	return "Hello World!"

@app.route('/retrieve', methods=['POST'])
def get_data():
	if request.method == 'POST':
		request_json = request.get_json(force=True)
		errors = validation.errors_login_params(request_json.keys())
		if errors:
			return errors
		return process_get(request_json['username'])
	return jsonify(error="invalid request type")

@app.route('/insert', methods=['POST'])
def insert_data():
	if request.method == 'POST':
		request_json = request.get_json(force=True)
		errors = validation.errors_insert_params(request_json.keys())
		if errors:
			return errors  
		username = request_json['username']
		password = request_json['password']
		if validation.login_check(username, password):
			for entry in request_json['data']:
				errors = validation.errors_insert_data_params(entry)
				if errors:
					return errors
				process_insert(username, entry['timestamp'], entry['gsr'], entry['hr'], entry['state'])
			return jsonify(success=str(len(request_json['data'])) + " entries inserted.")
		return jsonify(error="failed authentication")
	return jsonify(error="invalid request type")

def process_insert(username, timestamp, gsr, hr, state):
	start_db_request()
	cur = g.db.execute('insert into entries (userid, timestamp, hr, gsr, state, level) values (?, ?, ?, ?, ?, 0)', [username, timestamp, int(hr), int(gsr), state])
	g.db.commit()
	end_db_request()

def process_get(userid):
	start_db_request()
	cur = g.db.execute('select userid, timestamp, hr, gsr, state, level from entries where userid=?', (userid,))
	entries = [dict(username=row[0], timestamp=row[1], hr=row[2], gsr=row[3], state=row[4], level=row[5])for row in cur.fetchall()]
	end_db_request()
	return json.dumps([dict(item) for item in entries])

if __name__ == "__main__":
	app.run(host='159.203.31.236', debug=True)
	init_db()

