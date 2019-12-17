from gunicorn.app.base import BaseApplication
from blackbox.api.api import app
from blackbox import settings


class GunicornApp(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {
            key: value
            for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def main():
    options = {
        "bind": "{}:{}".format(settings.APP_HOST, settings.APP_PORT),
        "workers": 1,
        "threads": 12,
    }
    GunicornApp(app, options).run()


if __name__ == "__main__":
    main()
