import os
import json


root = 'DATASET/FF/Fake'
paths = os.walk(root)
new = []
for path, dir_list, file_list in paths:
    for file_name in file_list:
        if not file_name.endswith('.png'):
            continue
        new.append(os.path.join(path, file_name))
        if len(new) % 1000 == 0:
            print(len(new), path, file_name)

print (len(new))
json.dump(new ,open('DATASET/paths/paths_of_ff_fake.json', 'w'))

