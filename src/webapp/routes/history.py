import json
from flask import Blueprint, render_template
from flask_login import login_required, current_user
from src.webapp.models import Analysis

bp = Blueprint('history', __name__, template_folder='templates')

@bp.route('/')
@login_required
def index():
    records = Analysis.query.filter_by(user_id=current_user.id)\
                            .order_by(Analysis.timestamp.desc())\
                            .all()
    history = []
    for a in records:
        history.append({
          'time':     a.timestamp,
          'text':     a.text[:50] + ('…' if len(a.text)>50 else ''),
          'result':   a.result,
          'probas':   json.loads(a.proba_json)
        })
    return render_template('history.html', history=history)
