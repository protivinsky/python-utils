import os, re
from datetime import datetime
import unicodedata

os.environ['TEMP'] = 'D:/temp'


def logger(msg):
    print('{} -- {}'.format(datetime.now().strftime('%H:%M:%S.%f')[:10], msg))


def slugify(value, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)


def create_stamped_temp(dir=None):
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if dir is None:
        path = '{}/{}'.format(os.environ['TEMP'], stamp)
    else:
        path = '{}/{}/{}'.format(os.environ['TEMP'], dir, stamp)
    os.makedirs(path, exist_ok=True)
    return path

