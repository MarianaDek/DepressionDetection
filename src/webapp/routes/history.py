from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import login_required, current_user
from src.webapp.extensions import db
from src.webapp.models import Analysis

bp = Blueprint('history', __name__, template_folder='templates')

@bp.route('/history', methods=['GET'])
@login_required
def history():
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.timestamp.asc()).all()
    data = [{
        'id': a.id,
        'date': a.timestamp.strftime('%d-%m-%Y'),
        'time': a.timestamp.strftime('%H:%M'),
        'result': a.result,
        'probas': a.probabilities
    } for a in analyses]
    chart_labels = [f"{row['date']} {row['time']}" for row in data]
    chart_data = [row['probas'].get('Depressed', 0) * 100 for row in data]
    return render_template(
        'history.html',
        analyses=data,
        chart_labels=chart_labels,
        chart_data=chart_data
    )

@bp.route('/history/delete/<int:analysis_id>', methods=['POST'])
@login_required
def delete_analysis(analysis_id):
    a = Analysis.query.get_or_404(analysis_id)
    if a.user_id == current_user.id:
        db.session.delete(a)
        db.session.commit()
        flash('Analysis deleted', 'success')
    return redirect(url_for('history.history'))