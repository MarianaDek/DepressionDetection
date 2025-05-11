from flask import Blueprint, render_template, request, flash
from flask_login import login_required, current_user
from src.ml.inference import predict_affinity, CLASS_NAMES
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
        file = request.files.get('file')

        if file and file.filename:
            ext = file.filename.rsplit('.', 1)[1].lower()
            if ext == 'txt':
                text = read_txt(file.stream)
            elif ext == 'pdf':
                text = read_pdf(file.stream)
            elif ext in ('docx', 'doc'):
                text = read_docx(file.stream)
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
            label_idx, proba = predict_affinity(text)
            predicted_name = CLASS_NAMES[label_idx]
            probabilities = [
                (CLASS_NAMES[i], float(proba[i]))
                for i in range(len(proba))
            ]

            sentiment_scores = sia.polarity_scores(text)
            compound = sentiment_scores['compound']
            if compound >= 0.05:
                sentiment_label = 'Positive'
            elif compound <= -0.05:
                sentiment_label = 'Negative'
            else:
                sentiment_label = 'Neutral'

            sentiment_pct = round(compound * 50 + 50)

            record = Analysis(
                user_id=current_user.id,
                result=predicted_name,
                probabilities={name: p for name, p in probabilities},
                sentiment_score=compound,
                sentiment_label=sentiment_label
            )
            db.session.add(record)
            db.session.commit()

            return render_template(
                'result.html',
                predicted=predicted_name,
                probabilities=probabilities,
                sentiment_score=compound,
                sentiment_label=sentiment_label,
                sentiment_pct=sentiment_pct
            )
        except Exception as e:
            flash(f"Error: {e}", 'danger')
            return render_template('analyze.html')

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
    chart_data   = [row['probas'].get('Depression', 0) * 100 for row in data]

    return render_template(
        'history.html',
        analyses=data,
        chart_labels=chart_labels,
        chart_data=chart_data
    )