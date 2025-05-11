import os
from flask import Flask, request
from flask_migrate import Migrate
from flask_babel import Babel
from src.webapp.extensions import db, login_manager
from src.webapp.routes.routes import bp as webapp_bp
from src.webapp.routes.auth import bp as auth_bp
from src.webapp.models import User

def create_app():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    app = Flask(
        __name__,
        template_folder=os.path.join(BASE_DIR, 'src', 'webapp', 'templates'),
        static_folder=os.path.join(BASE_DIR, 'src', 'webapp', 'static')
    )


    app.config.from_object('config.Config')

    app.config['BABEL_DEFAULT_LOCALE'] = app.config.get('BABEL_DEFAULT_LOCALE', 'uk')
    app.config['BABEL_SUPPORTED_LOCALES'] = app.config.get('BABEL_SUPPORTED_LOCALES', ['uk', 'en'])


    def get_locale():
        lang = request.args.get('lang')
        if lang in app.config['BABEL_SUPPORTED_LOCALES']:
            return lang
        return app.config['BABEL_DEFAULT_LOCALE']


    babel = Babel(app, locale_selector=get_locale)
    app.jinja_env.add_extension('jinja2.ext.i18n')

    @app.context_processor
    def inject_locale():
        return dict(get_locale=get_locale)


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

    return app

if __name__ == '__main__':
    create_app().run(host='0.0.0.0', port=80, debug=True)
