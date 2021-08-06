import os
from importlib import import_module
from typing import Container

from dotenv import load_dotenv
import cv2
from flask import Flask, jsonify, request, make_response, Response
from flask_cors import CORS
from flask_pymongo import ObjectId

import datetime
import traceback

from inference import im
import helpers
from helpers.databases import db
import docker 

client = docker.from_env()

load_dotenv()

app = Flask(__name__)
CORS(app)


def send(data, status_code):
    """
    Sends out data from the app. Always use this to send out any kind of output within REST conventions.
    """
    return make_response(jsonify(data), status_code)


@app.route('/', methods=['GET'])
def check():
    """
    Returns an "alive" ping, with current system time and app name.
    """
    data = {
        "time": datetime.datetime.now(),
        "app": os.getenv("APP_NAME")
    }
    return send(data, 200)


@app.route('/<model_name>', methods=['GET'])
def get_model_info(model_name):
    """
    Wrapper function to get information about an inference model by passing its name in the URL.
    """
    try:
        model_data = im.model_info(model_name)
        return send(model_data, 200)
    except Exception as e:
        return send({
            "error": str(e)
        }, 404)


@app.route('/algorithm/<algorithm_id>/trigger/<function_name>', methods=['GET', 'POST'])
def algorithm_trigger_action(algorithm_id, function_name):
    """
    Wrapper function to trigger any method of any model
    """
    data = request.json
    print(algorithm_id, function_name)
    algorithm = db.TABLE_INFERENCE_ALGORITHMS.find_one({"_id": ObjectId(algorithm_id)})
    model_path = algorithm["path"]
    model_module_path = ".".join(["inference", "models", model_path, "methods"])
    module = import_module(model_module_path, package=model_path)
    func = getattr(module, function_name)
    return send(func(data), 200)


@app.route('/cameras', methods=['GET', 'POST'])
def camera_list():
    """
    Return the list of cameras
    """
    data = request.json
    print("data is ",data)
    if "company_id" in data.keys():
        db.set_company(data["company_id"])
    cameras = db.TABLE_CAMERAS.find({})
    output = []

    for camera in cameras:
        camera["_id"] = str(camera["_id"])
        output.append(camera)
    print("cameras",cameras)
    return send(output, 200)


@app.route('/company/<company_id>/camera/<camera_id>/live/raw', methods=['GET', 'POST'])
def camera_live_raw(company_id, camera_id):
    """
    Throw live raw feed of camera
    """
    #data = request.json
    #if "company_id" in data.keys():
    db.set_company(company_id)
    camera = db.TABLE_CAMERAS.find_one({"_id": ObjectId(camera_id)})
    camera_stream = cv2.VideoCapture(camera["cctv_feed_url"])
    return Response(helpers.raw_frames(camera_stream), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera/<camera_id>/live/inference', methods=['GET', 'POST'])
def camera_live_inference(camera_id):
    """
    Perform given inference model runs on given camera and throw live feed
    """
    data = request.json
    if "company_id" in data.keys():
        db.set_company(data["company_id"])
    camera = db.TABLE_CAMERAS.find_one({"_id": ObjectId(camera_id)})
    camera_company_id = camera["company_id"]
    camera_stream = cv2.VideoCapture(camera["cctv_feed_url"])
    camera_stream.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    camera_rois = db.TABLE_CAMERA_ROIS.find({"camera_id": ObjectId(camera_id)})
    rois = []
    extra_data = {}
    extra_data["company_id"] = camera_company_id
    extra_data["camera_id"] = camera["_id"]
    if "type" in camera.keys():
        extra_data["type"] = camera["type"]
    for camera_roi in camera_rois:
        rois.append(camera_roi)
    return Response(helpers.gen_inference_frames(camera_stream, rois, extra_data),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/docker/<docker_id>/start', methods=['GET', 'POST'])
def docker_start(docker_id):
    """
    Start docker container as per request
    """
    print(docker_id)
    container = client.containers.run(docker_id, detach = True)
    print(container.id)
    data = {
        "time": datetime.datetime.now(),
        "message": "Docker starting, please wait for further request atleast 5 min.",
        "container_id": container.id
    }
    return send(data, 200)


@app.route('/docker/<docker_id>/kill', methods=['GET', 'POST'])
def docker_kill(docker_id):
    """
    stop docker container as per request
    """
    print(docker_id)
    container = client.containers.get(docker_id)
    print(container)
    container.stop()
    data = {
        "time": datetime.datetime.now(),
        "message": "Docker killed",
        "container_id": container.id
    }
    return send(data, 200)


@app.route('/docker/<docker_id>/restart', methods=['GET', 'POST'])
def docker_restart(docker_id):
    """
    Start docker container as per request
    """
    print(docker_id)
    container = client.containers.get(docker_id)
    print(container)
    container.restart()
    print(container.id)
    data = {
        "time": datetime.datetime.now(),
        "message": "Docker restarting, please wait for further request atleast 5 min.",
        "container_id": container.id
    }
    return send(data, 200)


@app.route('/docker/<docker_id>/get', methods=['GET', 'POST'])
def docker_get(docker_id):
    """
    Start docker container as per request
    """
    print(docker_id)
    image = client.images.get(docker_id)
    print(image.id)
    data = {
        "time": datetime.datetime.now(),
        "message": "Docker restarting, please wait for further request atleast 5 min.",
        "image_id": image.id
    }
    return send(data, 200)


# Error Handler 404
@app.errorhandler(404)
def not_found(error):
    return make_response(error, 404)


# Error Handler 405
@app.errorhandler(405)
def method_not_allowed(error):
    return send({'error': 'Method is not allowed'}, 405)


# Error Handler 500
@app.errorhandler(500)
def internal_server_error(error):
    print(traceback.format_exc())
    return send({'error': 'Internal Server Error'}, 500)


# Exception
@app.errorhandler(Exception)
def unhandled_exception(error):
    print(traceback.format_exc())
    try:
        return make_response(error, 500)
    except:
        return send({'error': "Unknown error"}, 500)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
