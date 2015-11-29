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

@app.route('/get/<user_id>', methods=['GET'])
def get_data(user_id):
	if request.method == 'GET':
		return process_get(user_id);
		#errors = validation.check_get_request(request_json.keys())
		#if errors:
	#		return errors
	return jsonify(error="invalid request type")

@app.route('/insert', methods=['POST'])
def insert_data():
	if request.method == 'POST':
		request_json = request.get_json(force=True)
		errors = validation.check_insert_request(request_json.keys())
		if errors:
			return errors  
		username = request_json['username']
		password = request_json['password']
		if validation.login_check(username, password):
			timestamp = request_json['timestamp']
			gsr = request_json['gsr']
			hr = request_json['hr']
			return process_insert(username, timestamp, gsr, hr)
		return jsonify(error="failed authentication")
	return jsonify(error="invalid request type")

def process_insert(username, timestamp, gsr, hr):
	start_db_request()
	cur = g.db.execute('insert into entries (userid, timestamp, hr, gsr) values (?, ?, ?, ?)', [username, timestamp, int(hr), int(gsr)])
	g.db.commit()
	end_db_request()
	return jsonify(username=username, timestamp=timestamp, hr=hr, gsr=gsr)

def process_get(userid):
	start_db_request()
	result = ""
	cur = g.db.execute('select userid, timestamp, hr, gsr from entries where userid=?', (userid,))
	entries = [dict(user_id=row[0], timestamp=row[1], hr=row[2], gsr=row[3])for row in cur.fetchall()]
	end_db_request()
	return json.dumps([dict(item) for item in entries])

if __name__ == "__main__":
	app.run(host='159.203.31.236', debug=True)
	init_db()

