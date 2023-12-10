from gevent.pywsgi import WSGIServer
from app import app

if __name__ == '__main__':
    WSGIServer(('0.0.0.0', 5000), app).serve_forever()