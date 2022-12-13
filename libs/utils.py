import os
import tempfile
from datetime import datetime


os.environ['TEMP'] = 'D:/temp'


def logger(msg):
    print('{} -- {}'.format(datetime.now().strftime('%H:%M:%S.%f')[:10], msg))


def get_stamp():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def create_temp_dir(*args):
    tempdir = tempfile.gettempdir()
    path = os.path.join(tempdir, *args)
    os.makedirs(path, exist_ok=True)
    return path


def create_stamped_temp(*args):
    return create_temp_dir(*args, get_stamp())

