import socket
import threading
import os
import client_rec_aggr
import time
import buffer
import train_fl
import subprocess
import ssl
import security_config


HOST = '172.16.1.60'
PORT = 2003

ip_addr = '172.16.1.60'
local_port = '2025'

comm_rounds = 0
while comm_rounds < 2:
    print("=======Executing Client{} communication=======".format(comm_rounds))
    try:
        #train_fl.train_client(comm_rounds)
        # time.sleep(5)
        print("======Training successful============")

    except Exception as e:
        print("Exception occured while training:",e)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    clientname = 'Clients/C1'
    directory = '/home/014537055/FL_Project/TensorFlow-2.x-YOLOv3/checkpoints'

    salt = security_config.SALT
    key = security_config.getKey(security_config.getKDF(salt))
    fernet = security_config.getFernet(key)

    with s:
        sbuf = buffer.Buffer(s)
        sbuf.set_fernet(fernet)
        sbuf.put_bytes(salt)
        sbuf.put_utf8(ip_addr)
        sbuf.put_utf8(local_port)
        for file_name in os.listdir(directory):
            if "index" in file_name or "data" in file_name:
                print(directory+'/'+file_name)
                full_filename = directory+'/'+file_name
                sbuf.put_utf8(file_name)
                sbuf.put_utf8(clientname)

                # sbuf.put_utf8(ip_addr)

                # sbuf.put_utf8(local_port)
                file_size = os.path.getsize(full_filename)
                print('Send file: ', full_filename, ', size: ', file_size)
                #sbuf.put_utf8(str(file_size))

                sbuf.secure_send_file(full_filename)
                #with open(full_filename, 'rb') as f:
                #    sbuf.secure_put_bytes(f.read())
                print('File Sent')
        # s.close()    
    
    client_rec_aggr.rec_agg_weights(fernet)
    comm_rounds += 1
    print("comm rounds",comm_rounds)