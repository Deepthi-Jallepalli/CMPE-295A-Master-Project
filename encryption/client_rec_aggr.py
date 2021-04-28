import socket
import os
from _thread import *
import buffer
import time
import security_config

def rec_agg_weights(fernet):
    HOST = '0.0.0.0'
    PORT = 2025
    ThreadCount =0
    
    s = socket.socket()
    s.bind((HOST, PORT))
    s.listen(10)
    print("......Waiting for a connection.....")

    def multi_client(connbuf):
        count = 0
        while True:
            file_name = connbuf.get_utf8()
            if not file_name:
                break
            clientname = '/home/014537055/FL_Project/TensorFlow-2.x-YOLOv3/checkpoints'
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


'''def rec_agg_weights():
    HOST = '0.0.0.0'
    PORT = 2050
    
    s = socket.socket()
    s.bind((HOST, PORT))
    s.listen(10)
    print("......Waiting for a connection.....")
    count = 0
    while True:
        conn, addr = s.accept()
        print("Got a connection from ", addr)
        connbuf = buffer.Buffer(conn)
        file_name = connbuf.get_utf8()
        if not file_name:
            break
        clientname = '/home/014537055/FL_Project/TensorFlow-2.x-YOLOv3/checkpoints'

        while count <2:
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
            count +=1
        conn.close()

def rec_agg_weights():
    HOST = '0.0.0.0'
    PORT = 2050

    # If server and client run in same local directory,
    # need a separate place to store the uploads.
    # try:
    #     os.mkdir('uploads')
    # except FileExistsError:
    #     pass

    #comm_rounds = 0
    #while comm_rounds < 1:
    #ThreadCount =0
    
    s = socket.socket()
    s.bind((HOST, PORT))
    s.listen(10)
    print("......Waiting for a connection.....")
    #print("=======Executing Server{} communication=======".format(comm_rounds))

    def multi_client(connbuf):
        while True:
            # hash_type = connbuf.get_utf8()
            # if not hash_type:
            #     break
            # print('hash type: ', hash_type)

            file_name = connbuf.get_utf8()
            if not file_name:
                break
            #clientname = connbuf.get_utf8()
            clientname = '/home/014537055/FL_Project/TensorFlow-2.x-YOLOv3/checkpoints'
            file_name = os.path.join(clientname, file_name)
            # if not os.path.exists(directory):
            #     os.makedirs(directory)
            # file_name = os.path.join(directory,file_name)
            print('file name: ', file_name)
            # print('directory',directory)

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
        # print('Connection closed.')
        # conn.close()

    while True:
        conn, addr = s.accept()
        print("Got a connection from ", addr)
        connbuf = buffer.Buffer(conn)
        multi_client(connbuf)
        conn.close()
        break
        #ThreadCount += 1
            #print('Thread Number: ' + str(ThreadCount))

        #comm_rounds +=1


""" HOST = '0.0.0.0'
PORT = 2004

# If server and client run in same local directory,
# need a separate place to store the uploads.
# try:
#     os.mkdir('uploads')
# except FileExistsError:
#     pass

comm_rounds = 0
while  comm_rounds < 1:
    ThreadCount =0
    
    s = socket.socket()
    s.bind((HOST, PORT))
    s.listen(10)
    print("......Waiting for a connection.....")
    print("=======Executing Server{} communication=======".format(comm_rounds))

    def multi_client(connbuf):
        while True:
            # hash_type = connbuf.get_utf8()
            # if not hash_type:
            #     break
            # print('hash type: ', hash_type)

            file_name = connbuf.get_utf8()
            if not file_name:
                break
            clientname = connbuf.get_utf8()
            file_name = os.path.join(clientname, file_name)
            # if not os.path.exists(directory):
            #     os.makedirs(directory)
            # file_name = os.path.join(directory,file_name)
            print('file name: ', file_name)
            # print('directory',directory)

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
        # print('Connection closed.')
        # conn.close()

    while ThreadCount < 3:
        conn, addr = s.accept()
        print("Got a connection from ", addr)
        connbuf = buffer.Buffer(conn)
        start_new_thread(multi_client, (connbuf,))
        ThreadCount += 1
        print('Thread Number: ' + str(ThreadCount))

    comm_rounds +=1
 '''