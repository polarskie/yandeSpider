from urllib import request
import json
import time
import os
import ssl

MIN_WAIT = 20
TIMEOUT = 40000


class YandeExplorer:
    def __init__(self, exist_ids=(), save_dir="saved", tags=None):
        self.page_num = 0
        self.total_image_num = 0
        self.total_image_size = 0
        self.page_image_info_list = None
        self.index_on_page = 0
        self.saved_img_ids = set(exist_ids)
        self.fetched_img_ids = set(exist_ids)
        self.accessed_img_ids = set(exist_ids)
        self.save_dir = save_dir
        self.wait_before_next_request = MIN_WAIT
        self.parameters = ""
        if tags is not None:
            self.parameters = '&tags=' + '+'.join(tags)
        self.download_next_page()
            

    def wait(self):
        print("Waiting for %ss before next request" % self.wait_before_next_request)
        time.sleep(self.wait_before_next_request)
        if self.wait_before_next_request < 640:
            self.wait_before_next_request *= 2

    def download_next_page(self, attempts=0):
        pn = self.page_num + 1
        for _ in range(9999 if attempts <= 0 else attempts):
            try:
                with request.urlopen('https://yande.re/post?page=%i%s' % (pn, self.parameters), timeout=TIMEOUT, context=ssl._create_unverified_context()) as f:
                    if f.status != 200:
                        print("ERROR in download_next_page: https request failed \nstatus: ", f.status, " \nreason: ", f.reason)
                        self.wait()
                        continue
                    else:
                        self.wait_before_next_request = MIN_WAIT
                        data = f.read().decode('utf-8')
                        self.page_image_info_list = []
                        for line in data.split('\n'):
                            if line.strip().find('Post.register(') == 0:
                                info_raw = line.strip()[len('Post.register('):-1]
                                self.page_image_info_list.append(json.loads(info_raw))
                        # print('Status:', f.status, f.reason)
                        # for k, v in f.getheaders():
                        #     print('%s: %s' % (k, v))
                        # for info in self.page_image_info_list:
                        #     print(info['id'], info['file_url'])
                        self.page_num = pn
                        self.index_on_page = 0
                        return 0
            except:
                print("ERROR in download_next_page: time out.")
                print('https://yande.re/post?page=%i' % pn)
                self.wait()
        return None

    def fetch_next_raw(self, attempts=0, target_img='sample_url'):
        while True:
            if self.page_image_info_list is None or self.index_on_page == len(self.page_image_info_list):
                self.download_next_page(attempts=attempts)
            tmp_info = {**self.page_image_info_list[self.index_on_page], **{}}
            self.index_on_page += 1
            if tmp_info['id'] not in self.saved_img_ids:
                break
        for attempt_i in range(9999 if attempts <= 0 else attempts):
            try:
                start_time = time.time()
                with request.urlopen(tmp_info[target_img], timeout=TIMEOUT, context=ssl._create_unverified_context()) as f:
                    if f.status != 200:
                        print("ERROR in fetch_next_raw: https request failed \nstatus: ", f.status, " \nreason: ", f.reason)
                        self.wait()
                        continue
                    else:
                        # print("LOG in fetch_next_raw: urlopen cost %ss" % int(time.time() - start_time))
                        self.wait_before_next_request = MIN_WAIT
                        raw_img_data = f.read()
                        # print("LOG in fetch_next_raw: %ss elapsed after f.read()" % int(time.time() - start_time))
                        tmp_info['file_raw'] = raw_img_data
                        self.fetched_img_ids.add(tmp_info['id'])
                        self.accessed_img_ids.add(tmp_info['id'])
                        # print("LOG in fetch_next_raw: whole function cost %ss in the last request" % int(time.time() - start_time))
                        return tmp_info
            except:
                print("ERROR in fetch_next_raw: time out.")
                self.wait()
        return None

    def save_next(self, target_img='sample_url'):
        start_time = time.time()
        tmp_info = self.fetch_next_raw(target_img=target_img)
        with open('%s/%i_%s.%s' % (self.save_dir, tmp_info['id'], target_img, tmp_info[target_img].split('.')[-1]), 'wb') as f:
            f.write(tmp_info['file_raw'])
        with open('%s/%i.info' % (self.save_dir, tmp_info['id']), 'w') as f:
            f.write(json.dumps({i: j for i, j in tmp_info.items() if i != 'file_raw'}))
        self.saved_img_ids.add(tmp_info['id'])
        self.accessed_img_ids.add(tmp_info['id'])
        print("saved page %i, index %i, id %i, elapsed time %is" % (self.page_num, self.index_on_page, tmp_info['id'], int(time.time()-start_time)))


if __name__ == "__main__":
    save_dir = "rating_e"
    exists_id_list = [i[:i.rfind('.')] for i in os.listdir(save_dir)]
    # ye = YandeExplorer(exist_ids=exists_id_list, save_dir=save_dir, tags=['feet', 'order:random'])
    ye = YandeExplorer(exist_ids=exists_id_list, save_dir=save_dir, tags=['rating:e', 'order:random'])
    for i in range(20000):
        ye.save_next()

# with request.urlopen('https://yande.re/post?') as f:
#     data = f.read()
#     print('Status:', f.status, f.reason)
#     for k, v in f.getheaders():
#         print('%s: %s' % (k, v))
# with open("index.html", 'w', encoding='utf-8') as f:
#     f.write(data.decode('utf-8'))
