import os
from dotenv import load_dotenv

load_dotenv()
ACCESS_KEY = os.environ["ACCESS_KEY"]
SECRET_KEY = os.environ["SECRET_KEY"]
BUCKETNAME = os.environ["BUCKETNAME"]
REGION = os.environ["REGION"]