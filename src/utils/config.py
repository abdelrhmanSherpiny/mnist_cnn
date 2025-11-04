import tensorflow as tf
from dotenv import load_dotenv
import os

# load .env file
load_dotenv(override=True)

# Get the variables
APP_NAME = os.getenv("APP_NAME")
VERSION = os.getenv("VERSION")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

# Parent folder path
SRC_FOLDER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load Model
MODEL = tf.keras.models.load_model(os.path.join(SRC_FOLDER_PATH, "assets", "model.keras"))

CLASS_NAMES = ['T_Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_Boot']