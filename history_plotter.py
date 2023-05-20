import json
import os
import matplotlib

def plt_history():
    f = open(os.path.join(os.getcwd(), 'train_history.json'))
    data = json.load(f)
    print(data.keys())
    for key in data.keys():
        print(data[key])

plt_history()