import socket
import os
from _thread import *
import buffer
import server_aggegate_weights
import server_send_weights
import time

HOST = '0.0.0.0'
PORT = 3005

comm_rounds = 0
while  comm_rounds < 1:
    ThreadCount =0
    
    s = socket.socket()
    s.bind((HOST, PORT))
    s.listen(10)
    print("......Waiting for a connection.....")
    print("=======Executing Server{} communication=======".format(comm_rounds))
    hosts = []
    ports = []
    def multi_client(connbuf,hosts,ports):
        while True:
            file_name = connbuf.get_utf8()
            if not file_name:
                break
            clientname = connbuf.get_utf8()
            ip_address = connbuf.get_utf8()
            port = connbuf.get_utf8()
            hosts.append(ip_address)
            ports.append(int(port))
            file_name = os.path.join(clientname, file_name)
            print('client name:',clientname)
            print('client address and port',ip_address,port)
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
        start_new_thread(multi_client, (connbuf,hosts,ports))
        ThreadCount += 1
        print('Thread Number: ' + str(ThreadCount))
        if ThreadCount==4:
            time.sleep(10)
            break

    print('Starting Aggregation')
    server_aggegate_weights.aggregation()
    server_send_weights.send_agg_weights(hosts,ports)
    comm_rounds +=1
