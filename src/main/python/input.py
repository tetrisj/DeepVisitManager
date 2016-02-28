import numpy as np
import os
import simplejson
import itertools
from scipy import sparse
from pprint import pprint


def parse_event(s):
    data = simplejson.loads(s)

    feature_dict = {(0,int(k)): v for k, v in data['event']['feature'].items()}
    max_dim = max(feature_dict.keys())[1]+1
    print max_dim

    start_time = data['startTime']
    feature_dict[(0, max_dim-1)] -= start_time  # Adjust timestamp feature to be relative

    features = sparse.dok_matrix((1, max_dim))
    features.update(feature_dict)
    label = 1 if data['label'] else 0
    visit_id = data['visitId']

    return features, label, visit_id


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def event_data(root_path):
    for filename in os.listdir(root_path):
        if 'part' not in filename or 'crc' in filename:
            continue
        file_path = os.path.join(root_path, filename)
        print file_path
        current_visit = ''
        features = []
        labels = []
        for line in file(file_path, 'rb'):
            print line
            feature, label, visit_id = parse_event(line)
            if current_visit != visit_id and current_visit:
                yield sparse.vstack(features), np.array(labels)

                features = []
                labels = []
            features.append(feature.tocsr())
            labels.append(label)
            current_visit = visit_id


def main():
    labeled_input = event_data('/home/jenia/Deep/labeledJson')
    # for i in grouper(16,labeled_input).next():
    #     print i

    x = grouper(8, labeled_input).next()
    pprint(x)


if __name__ == '__main__':
    main()
