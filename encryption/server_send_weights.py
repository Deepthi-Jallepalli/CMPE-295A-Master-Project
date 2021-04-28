import socket
import threading
import os
import buffer

def send_agg_weights(hosts,ports,fernets):
    print("Sending agg weights")
    for h,p,f in zip(hosts,ports,fernets):
        print(h,p)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((h, p))
        directory = '/home/014537055/FL_Project/TensorFlow-2.x-YOLOv3/server_agg_weights'

        with s:
            
            sbuf = buffer.Buffer(s)
            sbuf.set_fernet(f)

            for file_name in os.listdir(directory):
                if "index" in file_name or "data" in file_name:
                    print(directory+'/'+file_name)
                    full_filename = directory+'/'+file_name
                    sbuf.put_utf8(file_name)
                    file_size = os.path.getsize(full_filename)
                    print('Send file: ', full_filename, ', size: ', file_size)
                    #sbuf.put_utf8(str(file_size))

                    sbuf.secure_send_file(full_filename)
                    #with open(full_filename, 'rb') as f:
                    #    sbuf.secure_put_bytes(f.read())
                    print('File Sent')
        
        s.close()
        print("File sent for Client")
