# src/webapp/routes/routes.py

import os
import joblib
import numpy as np
from flask import Blueprint, render_template, request, flash
from flask_login import login_required, current_user
from src.ml.inference import predict_full, BINARY_CLASS_NAMES, TYPE_CLASS_NAMES
from src.file_parsers import read_txt, read_pdf, read_docx
from src.webapp.extensions import db
from src.webapp.models import Analysis
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

bp = Blueprint('webapp', __name__,
               template_folder='templates',
               static_folder='static')


@bp.route('/', methods=['GET'])
def home():
    return render_template('home.html')


@bp.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        upload = request.files.get('file')
        if upload and upload.filename:
            ext = upload.filename.rsplit('.', 1)[-1].lower()
            if ext == 'txt':
                text = read_txt(upload.stream)
            elif ext == 'pdf':
                text = read_pdf(upload.stream)
            elif ext in ('docx', 'doc'):
                text = read_docx(upload.stream)
            else:
                flash('Unsupported file format', 'danger')
                return render_template('analyze.html')

        if not text:
            flash('Enter text or upload a file', 'warning')
            return render_template('analyze.html')
        words = text.split()
        if len(words) < 4:
            flash('Please enter at least 4 words', 'warning')
            return render_template('analyze.html')
        if len(set(words)) == 1:
            flash('Please enter different words.', 'warning')
            return render_template('analyze.html')

        try:
            age     = float(request.form['age'])
            gender  = float(request.form['gender'])
            age_cat = float(request.form['age_category'])
        except Exception:
            flash('Please fill in age, gender, and age category correctly', 'warning')
            return render_template('analyze.html')

        try:
            is_depr, prob_depr, type_label, type_proba = predict_full(
                text, age, gender, age_cat
            )
        except Exception as e:
            flash(f"Model error: {e}", 'danger')
            return render_template('analyze.html')

        sent = sia.polarity_scores(text)['compound']
        if sent >= 0.05:
            sent_label = 'Positive'
        elif sent <= -0.05:
            sent_label = 'Negative'
        else:
            sent_label = 'Neutral'
        sent_pct = round(sent * 50 + 50)

        result_label = 'Depressed' if is_depr else 'Not Depressed'
        binary_probs = {
            'Depressed': round(prob_depr, 4),
            'Not Depressed': round(1 - prob_depr, 4)
        }
        record = Analysis(
            user_id=current_user.id,
            result=result_label,
            probabilities={**binary_probs, **({'Type_'+type_label: float(type_proba.max())} if type_label else {})},
            sentiment_score=sent,
            sentiment_label=sent_label
        )
        db.session.add(record)
        db.session.commit()

        return render_template(
            'result.html',
            is_depressed=is_depr,
            prob_depressed=prob_depr,
            binary_probs=binary_probs,
            type_label=type_label,
            type_proba=type_proba,
            TYPE_CLASS_NAMES=TYPE_CLASS_NAMES,
            sentiment_score=sent,
            sentiment_label=sent_label,
            sentiment_pct=sent_pct
        )

    # GET
    return render_template('analyze.html')


@bp.route('/history', methods=['GET'])
@login_required
def history():
    analyses = Analysis.query \
        .filter_by(user_id=current_user.id) \
        .order_by(Analysis.timestamp.asc()) \
        .all()

    data = [{
        'timestamp': a.timestamp.strftime('%Y-%m-%d %H:%M'),
        'result':    a.result,
        'probas':    a.probabilities,
        'sentiment_score': a.sentiment_score,
        'sentiment_label': a.sentiment_label
    } for a in analyses]

    chart_labels = [row['timestamp'] for row in data]
    chart_data   = [row['probas'].get('Depressed', 0) * 100 for row in data]

    return render_template(
        'history.html',
        analyses=data,
        chart_labels=chart_labels,
        chart_data=chart_data
    )
