import json
import numpy


if __name__ == "__main__":
    with open("all_info_saved_1.info") as f:
        js = json.loads(f.read())
    tag_list_list = [info['tags'].split(' ') for info in js]
    statistical = {}
    for tag_list in tag_list_list:
        for tag in tag_list:
            if tag in statistical:
                statistical[tag] += 1
            else:
                statistical[tag] = 1
    statistical_list = [[k, v] for k, v in statistical.items()]
    statistical_list = sorted(statistical_list, key=lambda x : -x[1])
    print(statistical_list[:100])
    print(len(tag_list_list))
