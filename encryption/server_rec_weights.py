import socket
import os
from _thread import *
import buffer
import server_aggregate_weights
import server_send_weights
import time
import sys
import security_config
import threading

directory ='/home/014537055/FL_Project/TensorFlow-2.x-YOLOv3'
HOST = '0.0.0.0'
PORT = 2003
comm_rounds = 0
while  comm_rounds < 2:
    ThreadCount =0
    
    s = socket.socket()
    s.bind(( HOST, PORT))
    s.listen(10)
    print("......Waiting for a connection.....")
    print("=======Executing Server{} communication=======".format(comm_rounds))
    hosts = []
    ports = []
    fernets = []
    def multi_client(connbuf,hosts,ports,fernets):
        # Get 16 bytes of random salt to be used for this session
        salt = connbuf.get_bytes(16)
        print('-----------')
        print(salt)
        print(type(salt))
        key = security_config.getKey(security_config.getKDF(salt))
        fernet = security_config.getFernet(key)
        fernets.append(fernet)
        connbuf.set_fernet(fernet)
        print('key')
        print(key)
        print('key')
        
        ip_address = connbuf.get_utf8()
        print('RECV')
        print(type(ip_address))
        print(ip_address)
        port = connbuf.get_utf8()
        hosts.append(ip_address)
        ports.append(int(port))
        while True:

            # ip_address = connbuf.get_utf8()
            # port = connbuf.get_utf8()
            file_name = connbuf.get_utf8()
            if not file_name:
                break
            clientname = connbuf.get_utf8()
            # hosts.append(ip_address)
            # ports.append(int(port))
            print(hosts,ports)
            file_name = os.path.join(clientname, file_name)
            file_name = directory+'/'+file_name
            print('client name:',clientname)
            print('client address and port',ip_address,port)
            print('file name: ', file_name)
            file_size = int(connbuf.get_utf8())
            print('Encrypted file size: ', file_size)

            connbuf.secure_recv_file(file_size, file_name)

    thread_handles = []
    conns = []
    while True:
        conn, addr = s.accept()
        conns.append(conn)
        print("Got a connection from ", addr)
        connbuf = buffer.Buffer(conn)
        handle = threading.Thread(target=multi_client,args=(connbuf,hosts,ports,fernets))
        handle.start()
        thread_handles.append(handle)
        ThreadCount += 1
        print('Thread Number: ' + str(ThreadCount))
        if ThreadCount==1:
            time.sleep(10)
            for handle in thread_handles:
                handle.join()
            for conn in conns:
                conn.close()
            break
        

    print('Starting Aggregation')
    server_aggregate_weights.aggregation()
    server_send_weights.send_agg_weights(hosts,ports,fernets[0])
    comm_rounds +=1