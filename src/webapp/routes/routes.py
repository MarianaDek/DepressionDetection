import os
import joblib
import numpy as np
from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from src.ml.inference import predict_full, BINARY_CLASS_NAMES, TYPE_CLASS_NAMES
from src.file_parsers import read_txt, read_pdf, read_docx
from src.webapp.extensions import db
from src.webapp.models import Analysis

bp = Blueprint('webapp', __name__, template_folder='templates', static_folder='static')

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
            age = float(request.form['age'])
            gender = float(request.form['gender'])
            age_cat = float(request.form['age_category'])
        except (ValueError, KeyError):
            flash('Please fill in age, gender, and age category correctly', 'warning')
            return render_template('analyze.html')
        try:
            is_depr, prob_depr, type_label, type_proba = predict_full(text, age, gender, age_cat)
        except Exception as e:
            flash(f"Model error: {e}", 'danger')
            return render_template('analyze.html')

        result_label = 'Depressed' if is_depr else 'Not Depressed'
        binary_probs = {
            'Depressed': round(prob_depr, 4),
            'Not Depressed': round(1 - prob_depr, 4)
        }
        record = Analysis(
            user_id=current_user.id,
            result=result_label,
            probabilities={**binary_probs, **({'Type_' + type_label: float(type_proba.max())} if type_label else {})},
            sentiment_score=0.0,
            sentiment_label=''
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
        )
    return render_template('analyze.html')


@bp.route('/support/<disorder>')
@login_required
def support(disorder):
    support_data = {
        'Stress': {
            'title': 'Stress',
            'description': 'Information about stress and available support.',
            'resources': [
                {'name': 'Free online therapy sessions',
                 'url': 'https://www.therapyroute.com/free-therapy-for-ukraine'},
                {'name': 'Online support groups', 'url': 'https://www.heypeers.com/meetings/42423/details'},
                {'name': 'Articles about stress management', 'url': 'https://www.stress-management-articles.com'},
            ]
        },
        'Depression': {
            'title': 'Depression',
            'description': 'Information about depression and available support.',
            'resources': [
                {'name': 'Free online therapy sessions',
                 'url': 'https://www.therapyroute.com/free-therapy-for-ukraine'},
                {'name': 'Online support groups', 'url': 'https://www.heypeers.com/meetings/42410/details'},
                {'name': 'Articles about depression', 'url': 'https://www.depression-info.com'},
            ]
        },
        'Bipolar disorder': {
            'title': 'Bipolar disorder',
            'description': 'Information about bipolar disorder and available support.',
            'resources': [
                {'name': 'Free online therapy sessions',
                 'url': 'https://www.therapyroute.com/free-therapy-for-ukraine'},
                {'name': 'Online support groups', 'url': 'https://www.heypeers.com/meetings/41928/details'},
                {'name': 'Articles about bipolar disorder', 'url': 'https://www.bipolar-info.com'},
            ]
        },
        'Personality disorder': {
            'title': 'Personality disorder',
            'description': 'Information about personality disorders and available support.',
            'resources': [
                {'name': 'Free online therapy sessions',
                 'url': 'https://www.therapyroute.com/free-therapy-for-ukraine'},
                {'name': 'Online support groups', 'url': 'https://www.heypeers.com/meetings/41928/details'},
                {'name': 'Articles about personality disorders', 'url': 'https://www.personality-disorder-info.com'},
            ]
        },
        'Anxiety': {
            'title': 'Anxiety',
            'description': 'Information about anxiety and available support.',
            'resources': [
                {'name': 'Free online therapy sessions',
                 'url': 'https://www.therapyroute.com/free-therapy-for-ukraine'},
                {'name': 'Online support groups', 'url': 'https://www.heypeers.com/meetings/41928/details'},
                {'name': 'Articles about anxiety', 'url': 'https://www.anxiety-info.com'},
            ]
        },
    }
    data = support_data.get(disorder, {
        'title': disorder,
        'description': 'Information not available.',
        'resources': []
    })
    return render_template('support.html', disorder=data)
