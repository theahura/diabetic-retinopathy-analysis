import json
import os

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

import IPython


# Globals.
UPLOAD_FOLDER = './images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

LABEL_FP = './labels.json'

# Flask.
app = Flask(__name__, template_folder="frontend/",
            static_folder="frontend")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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
        print 'error: one'
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        print 'error: two'
        return redirect(request.url)
    if file and allowed_file(file.filename):
        print 'uploading'
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        i = 1
        while os.path.isfile(filepath):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'],
                                    str(i) + '_' + filename)
            i += 1
        print 'saving'
        file.save(filepath)

        labels = {'image-name': filepath}
        for k in request.form:
            labels[k] = request.form[k]

        with open(LABEL_FP, 'a') as f:
            f.write(json.dumps(labels) + '\n')

        return jsonify({'message': ''})


@app.route("/image/<path:fp>", methods=["GET"])
def get_image(fp):
    if fp and os.path.isfile(fp):
        fp = os.path.basename(fp)
        return send_from_directory(app.config['UPLOAD_FOLDER'], fp,
                                   mimetype='image/png')


def main():
    app.run(debug=False)


if __name__ == "__main__":
    main()
