from __future__ import division, print_function
import sys

import numpy as np 
import multiprocessing as mp, io 

import msgpack
import msgpack_numpy as mn
mn.patch()

def run(q, client, func, Xarr, Yarr):
    futures = []
    for x in Xarr:
        for y in Yarr:
            futures.append(client.submit(func, q, x, y=y))
    results = client.gather(futures)
    print(results)
    return results

def job(q, x, y=1):
    print(x,y)

    val = (x+y)**2
    q.put({'val':val})
    return val

def writer_process(stream_name, q):
    def writer(q):
        stream = io.open(stream_name,'wb', buffering=0)
        while True:
            data = q.get()
            stream.write(msgpack.packb(data))
    p = mp.Process(target=writer, args=(q, ))
    p.daemon = True
    p.start()
    #p.join()

def reader_process(filename):
    stream = io.open(filename,'rb')
    unpacker = msgpack.Unpacker()
    lines = 1
    data = []
    while True:
        buf = stream.read(1024)
        if not buf:
            break
        unpacker.feed(buf)
        for o in unpacker:
            lines +=1
            data.append(o)
    return data

if __name__ == '__main__':
    test = eval(sys.argv[1])
    
    if test:
        from distributed import Client
        client = Client()       ###### can this be done with scheduler / worker setup and still use 
                                ###### writer_process??
        stream_name = 'delete_me'
        q = mp.Manager().Queue()
        writer_process(stream_name, q)
        # q = None

        Xarr, Yarr = np.arange(10), np.arange(10)
        results = run(q, client, job, Xarr, Yarr)

    else:
        a = reader_process('delete_me')
        for x in a:
            print(x['val'])
