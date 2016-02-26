import numpy as np
from hashlib import md5
import os
import simplejson


def url_digest(url_str):
    # TODO: This is stupid. Change to something meningful
    if type(url_str) is unicode:
        url_str = url_str.encode('utf-8')
    if url_str.startswith('http://') or url_str.startswith('https://'):
        url_str = url_str.split('://', 1)[-1]
    if url_str.startswith('www.'):
        url_str = url_str[4:]
    url_str = url_str.rstrip('/')
    h = md5(url_str)
    return int(h.hexdigest()[:8], 16)


def parse_events(event_str):
    import simplejson

    data = simplejson.loads(event_str.strip())
    n = len(data['events'])
    Y = np.zeros(n)
    X = np.zeros((n, 4))
    accepted = set([url_digest(url) for url in data['pages']])
    for i, event in enumerate(data['events']):
        timestamp = event['timestamp']
        request_id = url_digest(event['requestUrl'])
        referrer_id = url_digest(event['referrerUrl'])
        prev_id = url_digest(event['prevUrl'])
        X[i] = np.array([timestamp, request_id, referrer_id, prev_id])
        if request_id not in accepted:
            Y[i] = 2  # not mine
    #TODO: separate domain from pages in id
    #TODO: start timestamps from zero
    return X, Y


def event_data(root_path):
    for filename in os.listdir(root_path):
        if 'part' not in filename:
            continue
        file_path = os.path.join(root_path, filename)
        for line in file(file_path, 'rb'):
            yield parse_events(line)


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def main():
    one_input = event_data(r'C:\Projects\Deep\combined').next()
    print one_input


if __name__ == '__main__':
    main()
