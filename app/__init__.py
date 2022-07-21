from flask import Flask

#ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)

#def allowed_file(filename):
#    return '.' in filename and \
#           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

from app import routes
