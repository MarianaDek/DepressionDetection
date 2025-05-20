import os
from flask import Flask, request, g, session
from flask_migrate import Migrate
from src.webapp.extensions import db, login_manager
from src.webapp.routes.routes import bp as webapp_bp
from src.webapp.routes.auth import bp as auth_bp
from src.webapp.routes.history import bp as history_bp
from src.webapp.models import User

from src.translations.translations import translations

DEFAULT_LOCALE = 'uk'
SUPPORTED_LOCALES = ['uk', 'en']

def create_app():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    app = Flask(
        __name__,
        template_folder=os.path.join(BASE_DIR, 'src', 'webapp', 'templates'),
        static_folder=os.path.join(BASE_DIR, 'src', 'webapp', 'static')
    )

    app.config.from_object('config.Config')

    app.config['DEFAULT_LOCALE'] = DEFAULT_LOCALE
    app.config['SUPPORTED_LOCALES'] = SUPPORTED_LOCALES


    @app.before_request
    def detect_locale():
        lang = request.args.get('lang')
        if lang and lang in app.config['SUPPORTED_LOCALES']:
            session['lang'] = lang
        lang = session.get('lang', app.config['DEFAULT_LOCALE'])
        g.lang = lang

    def _(text):
        lang = getattr(g, 'lang', app.config['DEFAULT_LOCALE'])
        return translations.get(lang, {}).get(text, text)

    @app.context_processor
    def inject_translations():
        return dict(_=_,
                    get_locale=lambda: getattr(g, 'lang', app.config['DEFAULT_LOCALE']))

    db.init_app(app)
    Migrate(app, db)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    with app.app_context():
        db.create_all()

    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(webapp_bp)
    app.register_blueprint(history_bp, url_prefix='/history')

    return app


if __name__ == '__main__':
    create_app().run(host='0.0.0.0', port=80, debug=True)
