#================================================================
#
#   File name   : server_send_weights.py
#   Description : Socket Programming to send weights to the server
#
#================================================================
import socket
import threading
import os
import buffer

def send_agg_weights(hosts,ports,fernets):
    print("========== Server sending agg weights ==========\n")
    for h,p,f in zip(hosts,ports,fernets):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((h, p))
        directory = '/home/014505660/fl_project/TensorFlow-2.x-YOLOv3/server_agg_weights'

        with s:
            sbuf = buffer.Buffer(s)
            sbuf.set_fernet(f)
            print("========== Iniating connection for Client with host {} and port {} \n".format(h,p))
            for file_name in os.listdir(directory):
                if "index" in file_name or "data" in file_name:
                    print('File Path',directory+'/'+file_name,'\n')
                    full_filename = directory+'/'+file_name
                    sbuf.put_utf8(file_name)
                    file_size = os.path.getsize(full_filename)
                    print('Send file: ', full_filename, ', size: ', file_size,'\n')
                    sbuf.secure_send_file(full_filename)
                    print('File Sent successfully \n')
        
        s.close()
        print("Weights Files sent to Client successfully \n")