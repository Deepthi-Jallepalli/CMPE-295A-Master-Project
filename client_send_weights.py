import socket
import threading
import os
import client_rec_aggr

import buffer
import train

HOST = '172.16.1.210'
PORT = 3005

ip_addr = '172.16.1.207'
local_port = '2009'

comm_rounds = 0
while comm_rounds < 1:
    print("=======Executing Client{} communication=======".format(comm_rounds))
    try:

        train.train_client(comm_rounds)
        print("======Training successful============")

    except Exception as e:
        print("Exception occured while training:",e)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    clientname = 'Clients/C4'
    directory = '/home/014489241/fl_project/TensorFlowYOLOv3/checkpoints'

    with s:
        
        sbuf = buffer.Buffer(s)
        for file_name in os.listdir(directory):
            if "index" in file_name or "data" in file_name:
                print(directory+'/'+file_name)
                full_filename = directory+'/'+file_name
                sbuf.put_utf8(file_name)

                sbuf.put_utf8(clientname)

                sbuf.put_utf8(ip_addr)

                sbuf.put_utf8(local_port)
                file_size = os.path.getsize(full_filename)
                sbuf.put_utf8(str(file_size))

                with open(full_filename, 'rb') as f:
                    sbuf.put_bytes(f.read())
                print('File Sent')
        
    
    client_rec_aggr.rec_agg_weights()
    comm_rounds += 1