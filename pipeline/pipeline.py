import os

import cv2
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory
import numpy as np
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
        im, preds, hms = model_runner.get_prediction(filepath)
        hm_fps = []
        hm_im_fps = []
        print 'got output'
        for i in range(0, model.NUM_LABELS):
            hm = cv2.applyColorMap(np.uint8(hms[i][0]*255), cv2.COLORMAP_JET)
            hm_fp = '%s_hm_%d.png' % (os.path.splitext(filepath)[0], i)
            hm_fps.append(hm_fp)
            cv2.imwrite(hm_fp, hm)
            hm_im = hm * 0.5 + im * 0.5
            hm_im_fp = '%s_hm_process%d.png' % (
                os.path.splitext(filepath)[0], i)
            hm_im_fps.append(hm_im_fp)
            cv2.imwrite(hm_im_fp, hm_im)
        im_fp = os.path.splitext(filepath)[0] + '_processed.png'
        cv2.imwrite(im_fp, im)
        print 'saved output'
        resp = {'hm': hm_fps, 'pred': preds[0].tolist(), 'im_p': im_fp,
                'hm_im': hm_im_fps}
        return jsonify(resp)


@app.route("/image/<path:fp>", methods=["GET"])
def get_image(fp):
    if fp and os.path.isfile(fp):
        fp = os.path.basename(fp)
        return send_from_directory(app.config['UPLOAD_FOLDER'], fp,
                                   mimetype='image/png')


def main():
    app.run(debug=True)


if __name__ == "__main__":
    main()
