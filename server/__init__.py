from flask import Flask

def create_app():
    app = Flask(__name__)

    # Import routes after creating app to avoid circular imports
    from .routes import setup_routes
    setup_routes(app)

    return app
