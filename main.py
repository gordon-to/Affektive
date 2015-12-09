import sqlite3
from datetime import datetime
import flask.ext.restless
from flask import Flask, request, jsonify, session, g, redirect, url_for, abort, render_template, flash
from models import db, Measurement
from contextlib import closing
import validation

# configuration
DATABASE = 'affektive.db'
DEBUG = True
SECRET_KEY = 'development key'
USERNAME = 'admin'
PASSWORD = 'default'


def create_app():
	app = Flask(__name__)
	app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stress.db'
	app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
	db.init_app(app)
	with app.app_context():
		db.create_all()
	return app

app = create_app()

# Create API endpoints, which will be available at /api/<tablename> by
# default. Allowed HTTP methods can be specified as well.
def create_api(app):
	with app.app_context():
		manager = flask.ext.restless.APIManager(app, flask_sqlalchemy_db=db)
		manager.create_api(Measurement, methods=['GET', 'PATCH' 'POST', 'DELETE'], res)
create_api(app)



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
		return jsonify(error=str(e)), 500




@app.route('/')
def main():
	return "Hello World!"

if __name__ == "__main__":
	app.run(host='localhost', debug=True)

