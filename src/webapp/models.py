from datetime import datetime
from flask_login import UserMixin
from src.webapp.extensions import db

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    analyses      = db.relationship('Analysis', backref='user', lazy=True)

class Analysis(db.Model):
    __tablename__ = 'analyses'
    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    timestamp        = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    result           = db.Column(db.String(64), nullable=False)
    probabilities    = db.Column(db.JSON, nullable=False)

    sentiment_score  = db.Column(db.Float, nullable=False)
    sentiment_label  = db.Column(db.String(16), nullable=False)

