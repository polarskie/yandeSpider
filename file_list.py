import json
import numpy as np
import time
import os


file_exts = set()
target_tag = "yukata"
train_portion = 0.9
positive_ids = [fn.split('.')[0] for fn in os.listdir(target_tag) if fn[-4:] == "info"]
positive_id_2_fn = {fn.split('_')[0]:fn for fn in os.listdir(target_tag) if fn[-4:] != "info"}
np.random.seed(int(time.time()))
with open("all_info_saved_1.info") as f:
    info_list = json.loads(f.read())
id_list = [str(info["id"]) for info in info_list]
tags_list = [info["tags"].strip().split(' ') for info in info_list]
negative_inds = [i for i, tags in enumerate(tags_list) if target_tag not in tags]
negative_ids = [id_list[i] for i in negative_inds]
negative_id_2_fn = {fn.split('_')[0]:fn for fn in os.listdir("saved_1") if fn[-4:] != "info"}
file_exts = file_exts | set([fn.split('.')[-1] for fn in positive_id_2_fn.values()])
file_exts = file_exts | set([fn.split('.')[-1] for fn in negative_id_2_fn.values()])

cat_sample_num = min(len(positive_ids), len(negative_ids))
np.random.shuffle(positive_ids)
np.random.shuffle(negative_ids)
positive_ids = positive_ids[:cat_sample_num]
negative_ids = negative_ids[:cat_sample_num]
split_ind = int(train_portion * cat_sample_num)
train_pos_ids = positive_ids[:split_ind]
test_pos_ids = positive_ids[split_ind:]
train_neg_ids = negative_ids[:split_ind]
test_neg_ids = negative_ids[split_ind:]
lines = []
with open("%s_train_list.txt" % target_tag, 'w') as f:
    for i in train_pos_ids:
        if i in positive_id_2_fn and positive_id_2_fn[i][-3:] == 'jpg':
            lines.append("%s/%s %i" % (target_tag, positive_id_2_fn[i], 1))
    for i in train_neg_ids:
        if i in negative_id_2_fn and negative_id_2_fn[i][-3:] == 'jpg':
            lines.append("saved_1/%s %i" % (negative_id_2_fn[i], 0))
    np.random.shuffle(lines)
    print(len(lines))
    f.write('\n'.join(lines))
lines = []
with open("%s_test_list.txt" % target_tag, 'w') as f:
    for i in test_pos_ids:
        if i in positive_id_2_fn and positive_id_2_fn[i][-3:] == 'jpg':
            lines.append("%s/%s %i" % (target_tag, positive_id_2_fn[i], 1))
    for i in test_neg_ids:
        if i in negative_id_2_fn and negative_id_2_fn[i][-3:] == 'jpg':
            lines.append("saved_1/%s %i" % (negative_id_2_fn[i], 0))
    np.random.shuffle(lines)
    print(len(lines))
    f.write('\n'.join(lines))

