from flask import Flask
from flask_pymongo import PyMongo
import os
from dotenv import load_dotenv
import redis

load_dotenv()
app = Flask(__name__)


class DB:
    def __init__(self):
        self.visioncloud_client = PyMongo(app, os.getenv("VISIONCLOUD_MONGO_URI"))
        self.contavision_client = PyMongo(app, os.getenv("CONTAVISION_MONGO_URI"))
        self.redis_client = redis.Redis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"))

        self.TABLE_CAMERAS = self.visioncloud_client.db.camera_si31793
        self.TABLE_EMPLOYEES = self.visioncloud_client.db.employee_si31793
        self.TABLE_CAMERA_ROIS = self.contavision_client.db.camera_rois
        self.TABLE_INSTANCES = self.contavision_client.db.instances
        self.TABLE_INFERENCE_ALGORITHMS = self.contavision_client.db.inference_algorithms
        self.TABLE_ATTENDANCE = self.contavision_client.db.attendances

    def set_company(self, company_id):
        self.TABLE_CAMERAS = getattr(self.visioncloud_client.db, "camera_" + str(company_id).lower())
        self.TABLE_EMPLOYEES = getattr(self.visioncloud_client.db, "employee_" + str(company_id).lower())


db = DB()
