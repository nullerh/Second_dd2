import json
import os
import matplotlib
import numpy as np

def plt_history():
    f = open('D:/predictions/train_history.json')
    # f = open(os.path.join(os.getcwd(), 'train_history.json'))
    data = json.load(f)
    print(data.keys())
    f1_scores = {'arousal': .0, 'leg movement': .0, 'Sleep-disordered breathing': .0}
    for key in data.keys():
        print(key)
        print(data[key])

        '''else:
            print(data[key].keys())
            print(data[key]['validation'])'''
        #print(f1_scores)


plt_history()