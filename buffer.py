#================================================================
#
#   File name   : buffer.py
#   Description : Performs read and write operation for weights transfer
#
#================================================================
import security_config
import os

class Buffer:
    def __init__(self,s):
        '''Buffer a pre-created socket.
        '''
        self.sock = s
        self.buffer = b''
        self.fernet = None

    def set_fernet(self, f):
        self.fernet = f

    def get_bytes(self,n):
        '''Read exactly n bytes from the buffered socket.
           Return remaining buffer if <n bytes remain and socket closes.
        '''
        counter = 0
        while len(self.buffer) < n:
            data = self.sock.recv(1024)
            counter +=1
            if not data:
                print("Packects recv:",counter)
                data = self.buffer
                self.buffer = b''
                return data
            self.buffer += data
        # split off the message bytes from the buffer.
        data,self.buffer = self.buffer[:n],self.buffer[n:]
        # print("Packects recv:",counter)
        return data
    
    def secure_recv_file(self, file_size, file_name):
        with open(file_name+'_enc', 'wb') as f:
            remaining = file_size
            while remaining:
                chunk_size = 4096 if remaining >= 4096 else remaining
                each_chunk = self.get_bytes(chunk_size)
                if not each_chunk: break
                f.write(each_chunk)
                remaining -= len(each_chunk)
            if remaining:
                print('----------------------File incomplete.  Missing',remaining,'bytes.')
                raise ValueError('Receive file failed')
            else:
                #Have to close here to flush data to disk
                f.close()
                if os.path.getsize(file_name+'_enc') != file_size:
                    print('Encrypted file size:', os.path.getsize(file_name+'_enc'), ', is not the same as received size:', file_size)
                    raise ValueError('File is incomplete')
                with open(file_name+'_enc', 'rb') as fr:
                    with open(file_name, 'wb') as fw:
                        data = fr.read()
                        recv_file_buff = security_config.decrypt(self.fernet, data)
                        fw.write(recv_file_buff)
                        fw.close()
                    fr.close()
                    print('File received successfully. size: ', os.path.getsize(file_name))
            os.remove(file_name+'_enc')

    def put_bytes(self,data):
        self.sock.sendall(data)

    def secure_send_file(self,file_name):
        if self.fernet == None:
            raise ValueError('Fernet object is null)')
        with open(file_name, 'rb') as f:
            data = f.read()
            cipherText = security_config.encrypt(self.fernet, data)
            print('Send encrypted file size = ', len(cipherText))
            self.put_utf8(str(len(cipherText)))
            self.sock.sendall(cipherText)

    def get_utf8(self):
        '''Read a null-terminated UTF8 data string and decode it.
           Return an empty string if the socket closes before receiving a null.
        '''
        while b'\x00' not in self.buffer:
            data = self.sock.recv(1024)
            if not data:
                return ''
            self.buffer += data
        # split off the string from the buffer.
        data,_,self.buffer = self.buffer.partition(b'\x00')
        if self.fernet == None:
            print('Received plain text')
            return data.decode()
        else:
            return security_config.decrypt(self.fernet, data).decode()

    def put_utf8(self,data):
        if '\x00' in data:
            raise ValueError('string contains delimiter(null)')
        if self.fernet == None:
            print('Send plain text')
            self.sock.sendall(data.encode() + b'\x00')
        else:
            cipherText = security_config.encrypt(self.fernet, data.encode())
            self.sock.sendall(cipherText + b'\x00')