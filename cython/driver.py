from __future__ import division, print_function
import sys

import numpy as np 

import risk_scaling as rs 
from utilsMetric import *

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