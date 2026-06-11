from flask import Flask
from config import Config
from api.health_routes import health_bp
from api.chat_routes import chat_bp
from db.sqlite_init import init_sqlite
from db.chroma_init import init_chroma

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    init_sqlite()
    init_chroma()
    app.register_blueprint(health_bp)
    app.register_blueprint(chat_bp)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=Config.PORT, debug=Config.DEBUG)