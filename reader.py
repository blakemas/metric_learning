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

if __name__=='__main__':
    import sys
    print reader_process(sys.argv[1])
