import socket
import os
from _thread import *
import buffer

def rec_agg_weights():
    HOST = '0.0.0.0'
    PORT = 2009
    ThreadCount =0
    
    s = socket.socket()
    s.bind((HOST, PORT))
    s.listen(10)
    print("......Waiting for a connection.....")

    def multi_client(connbuf):
        while True:
            file_name = connbuf.get_utf8()
            if not file_name:
                break
            clientname = '/home/014489241/fl_project/TensorFlowYOLOv3/checkpoints'
            file_name = os.path.join(clientname, file_name)
            print('file name: ', file_name)
            file_size = int(connbuf.get_utf8())
            print('file size: ', file_size )

            with open(file_name, 'wb') as f:
                remaining = file_size
                while remaining:
                    chunk_size = 4096 if remaining >= 4096 else remaining
                    chunk = connbuf.get_bytes(chunk_size)
                    if not chunk: break
                    f.write(chunk)
                    remaining -= len(chunk)
                if remaining:
                    print('File incomplete.  Missing',remaining,'bytes.')
                else:
                    print('File received successfully.')

    while True:
        conn, addr = s.accept()
        print("Got a connection from ", addr)
        connbuf = buffer.Buffer(conn)
        multi_client(connbuf)