# examples/1_hello/hello.py
from MyJapronto import MyApplication
import asyncio
import time
import numpy as np
import json
import os
import pickle


logs = [{}]


async def get_js(request):
    fn = request.match_dict['p']
    if '/' in fn or '.js' != fn.strip()[-3:]:
        return request.Response(code=404)
    try:
        with open("js/%s" % fn, 'r') as f:
            file_str = f.read()
    except:
        return request.Response(code=404)
    l = len(file_str)
    request.transport.set_write_buffer_limits(high=l * 4)
    return request.Response(text=file_str, mime_type="application/x-javascript")


async def log_server(request):
    if request.remote_addr not in ["0.0.0.0", "127.0.0.1", "localhost"]:
        return request.Response(code=404)
    logs[0] = json.loads(request.body.decode('utf-8'))
    return request.Response(code=200)


async def get_log_server(request):
    for k, v in logs[0].items():
        for i in np.reshape(np.argwhere(np.isnan(v)), [-1]):
            v[i] = v[i-1]
    return request.Response(body=json.dumps(logs[0]).encode('utf-8'))


async def show():
    while True:
        print(logs[0])
        await asyncio.sleep(10)

async def board_server(request):
    with open("board.html", 'r') as f:
        file_str = f.read()
    return request.Response(text=file_str, mime_type='text/html')

app = MyApplication()
# app.add_task(show())
app.router.add_route('/', board_server)
app.router.add_route('/log', log_server)
app.router.add_route('/get_log', get_log_server)
app.router.add_route('/js/{p}', get_js)
# app.router.add_route('/verify', verify)
app.run(debug=True, port=8080)
print("exit")
