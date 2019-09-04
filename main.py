import settings
from blackbox.api.api import app

if __name__ == '__main__':
    app.run(host=settings.APP_HOST, port=settings.APP_PORT, debug=settings.APP_DEBUG)
