import socket
import threading
import os
import client_rec_aggr
import time
import buffer
import train
import subprocess
import ssl
import security_config


HOST = '172.16.1.209'
PORT = 2003

ip_addr = '172.16.1.202'
local_port = '2004'

comm_rounds = 0
while comm_rounds < 1:
    print("======= Executing Client side round {} communication======= \n".format(comm_rounds))
    try:
        train.train_client(comm_rounds)
        time.sleep(5)
        print("======= Training completed successfully ============")

    except Exception as e:
        print("Exception occured while training:",e)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    clientname = 'Clients/C4'
    directory = '/home/014489241/fl_project/TensorFlowYOLOv3/checkpoints'

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
                print("File path ", directory+'/'+file_name, '\n')
                full_filename = directory+'/'+file_name
                sbuf.put_utf8(file_name)
                sbuf.put_utf8(clientname)

                # sbuf.put_utf8(ip_addr)

                # sbuf.put_utf8(local_port)
                file_size = os.path.getsize(full_filename)
                print('Send file: ', full_filename, ', size: ', file_size, '\n')
                #sbuf.put_utf8(str(file_size))

                sbuf.secure_send_file(full_filename)
                #with open(full_filename, 'rb') as f:
                #    sbuf.secure_put_bytes(f.read())
                print('File encrypted and sent successfully')
        # s.close()    
    
    client_rec_aggr.rec_agg_weights(fernet)
    comm_rounds += 1
    print(" executed communication rounds",comm_rounds,'\n')
