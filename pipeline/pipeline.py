import os

from flask import Flask, jsonify, redirect, render_template, request
from werkzeug.utils import secure_filename

import model
import IPython


# Globals.
UPLOAD_FOLDER = './images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Flask.
app = Flask(__name__, template_folder="frontend/webportal/",
            static_folder="frontend/webportal")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tensorflow.
model_runner = model.Runner()

# Helpers.
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Routes.
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    print 'got request'
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        i = 1
        while os.path.isfile(filepath):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'],
                                    str(i) + '_' + filename)
            i += 1
        file.save(filepath)
        return str(model_runner.get_prediction(filepath))


def main():
    app.run(debug=False)


if __name__ == "__main__":
    main()
