#================================================================
#
#   File name   : client_rec_aggr.py
#   Description : Receives encrypted aggregated weights from the server
#
#================================================================
import socket
import os
from _thread import *
import buffer
import time
import security_config

def rec_agg_weights(fernet):
    HOST = '0.0.0.0'
    PORT = 2005
    ThreadCount =0
    
    s = socket.socket()
    s.bind((HOST, PORT))
    s.listen(10)
    print("==================== Client is listening =================\n")

    def multi_client(connbuf):
        count = 0
        while True:
            file_name = connbuf.get_utf8()
            if not file_name:
                break
            clientname = '/home/014505660/fl_project/TensorFlow-2.x-YOLOv3/checkpoints'
            file_name = os.path.join(clientname, file_name)
            print('file name: ', file_name)
            file_size = int(connbuf.get_utf8())
            print('-------------Encrypted file size: ', file_size )

            connbuf.secure_recv_file(file_size, file_name)

    while True:
        conn, addr = s.accept()
        print("Got a connection from ", addr)
        connbuf = buffer.Buffer(conn)
        connbuf.set_fernet(fernet)
        multi_client(connbuf)
        print("Done receiving")
        conn.close()
        time.sleep(15)
        break