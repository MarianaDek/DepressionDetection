from flask import Blueprint, render_template, request, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required

from src.webapp.extensions import db
from src.webapp.models     import User

bp = Blueprint('auth', __name__,
               template_folder='templates/auth',
               static_folder='../static')

@bp.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        email    = request.form['email']
        password = request.form['password']
        if User.query.filter_by(email=email).first():
            flash('This email is already registered.', 'warning')
        else:
            user = User(
                email=email,
                password_hash=generate_password_hash(password)
            )
            db.session.add(user)
            db.session.commit()
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('auth.login'))
    return render_template('register.html')

@bp.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email    = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('webapp.home'))
        flash('Incorrect email or password', 'danger')
    return render_template('login.html')

@bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You are logged out.', 'info')
    return redirect(url_for('webapp.home'))
