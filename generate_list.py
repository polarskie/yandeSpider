import os
import numpy as np
import time
import json


neg_fns = os.listdir("saved_1")
neg_ids = []
for fn in neg_fns:
    if fn[-4:] == "info":
        with open("saved_1/%s" % fn, 'r') as f:
            tmp = json.loads(f.read())
            if tmp["rating"] != 'e':
                neg_ids.append(str(tmp["id"]))
neg_id_set = set(neg_ids)
neg_id_2_path = {fn.split('_')[0]:"saved_1/"+fn for fn in neg_fns if fn[-4:]!="info" and fn[-3:]=="jpg" and fn.split('_')[0] in neg_id_set}

pos_id_2_path = {fn.split('_')[0]:"rating_e/"+fn for fn in os.listdir("rating_e") if fn[-3:]=="jpg"}

np.random.seed(3897342)
neg_keys = list(neg_id_2_path.keys())
pos_keys = list(pos_id_2_path.keys())
cn = min(len(neg_keys), len(pos_keys))
tn = int(cn * 0.95)
np.random.shuffle(neg_keys)
np.random.shuffle(pos_keys)
neg_keys = neg_keys[:cn]
pos_keys = pos_keys[:cn]
train_pos_lines = [pos_id_2_path[k] + " 1" for k in pos_keys[:tn]]
train_neg_lines = [neg_id_2_path[k] + " 0" for k in neg_keys[:tn]]
test_pos_lines = [pos_id_2_path[k] + " 1" for k in pos_keys[tn:]]
test_neg_lines = [neg_id_2_path[k] + " 0" for k in neg_keys[tn:]]
print(len(train_pos_lines))
print(len(train_neg_lines))
print(len(test_pos_lines))
print(len(test_neg_lines))
with open("rating_train_list.txt", "w") as f:
    f.write("\n".join(train_pos_lines + train_neg_lines))

with open("rating_test_list.txt", "w") as f:
    f.write("\n".join(test_pos_lines + test_neg_lines))
