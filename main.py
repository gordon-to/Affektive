import sqlite3
from datetime import datetime
import flask.ext.restless
from flask import Flask, request, jsonify, session, g, redirect, url_for, abort, render_template, flash
from flask_sqlalchemy import SQLAlchemy
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
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stress.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Measurement(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	userid = db.Column(db.Integer)
	timestamp = db.Column(db.DateTime)
	hr = db.Column(db.Integer)
	gsr = db.Column(db.Float)
	state = db.Column(db.String(80))
	level = db.Column(db.Float)

	def __init__(self, userid, timestamp, hr, gsr, state, level):
		self.userid = userid
		self.timestamp = timestamp
		self.hr = hr
		self.gsr = gsr
		self.state = state
		self.level = level
		

	def __repr__(self):
		return '<Measurement state:{0}, hr:{1}, gsr:{2} '.format(self.state, self.hr, self.gsr)

db.create_all()

# Create API endpoints, which will be available at /api/<tablename> by
# default. Allowed HTTP methods can be specified as well.
manager = flask.ext.restless.APIManager(app, flask_sqlalchemy_db=db)
manager.create_api(Measurement, methods=['GET', 'POST', 'DELETE'])

@app.route('/measurements_batch', methods=['POST'])
def measruements_batch():
	try:
		json = request.get_json()
		userid = json['username']
		for entry in json['measurements']:
			measurement = Measurement(
			 	int(userid), 
			 	datetime.fromtimestamp(entry['timestamp']),
			 	int(entry['hr']),
			 	float(entry['gsr']),
				entry['state'],
				float(entry['level']))
			db.session.add(measurement)
		db.session.commit()
		return jsonify(success=str(len(json['measurements'])) + " entries inserted.")
	except Exception, e:

		return jsonify(error=str(e), )




@app.route('/')
def main():
	return "Hello World!"

if __name__ == "__main__":
	app.run(host='localhost', debug=True)

