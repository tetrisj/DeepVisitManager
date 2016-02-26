import numpy as np
import os
import simplejson
import itertools

def parse_event(s):
    data = simplejson.loads(s)
    feature = data['event']['feature']
    label = 1 if data['label'] else 0
    visit_id = data['visitId']
    return feature, label, visit_id


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def event_data(root_path):
    for filename in os.listdir(root_path):
        if 'part' not in filename:
            continue
        file_path = os.path.join(root_path, filename)
        current_visit = ''
        features = []
        labels = []
        for line in file(file_path, 'rb'):
            feature, label, visit_id = parse_event(line)
            if current_visit != visit_id and current_visit:
                yield features,labels
            features.append(feature)
            labels.append(label)
            current_visit = visit_id


def main():
    input = event_data(r'/home/jenia/Deep/labeledJson')
    # for i in grouper(16,input).next():
    #     print i
    from pprint import pprint
    x = grouper(8,input).next()


if __name__ == '__main__':
    main()
