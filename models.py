from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

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