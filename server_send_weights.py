import socket
import threading
import os
import buffer

def send_agg_weights(hosts,ports):
    print("Sending agg weights")
    for h,p in zip(hosts,ports):
        print(h,p)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((h, p))
        directory = '/home/014489241/fl_project/TensorFlowYOLOv3/server_agg_weights'

        with s:
            
            sbuf = buffer.Buffer(s)

            for file_name in os.listdir(directory):
                if "index" in file_name or "data" in file_name:
                    print(directory+'/'+file_name)
                    full_filename = directory+'/'+file_name
                    sbuf.put_utf8(file_name)
                    file_size = os.path.getsize(full_filename)
                    sbuf.put_utf8(str(file_size))

                    with open(full_filename, 'rb') as f:
                        sbuf.put_bytes(f.read())
                    print('File Sent')
        
        s.close()
        print("File sent for Client")        
            