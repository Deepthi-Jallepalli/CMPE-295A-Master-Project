#================================================================
#
#   File name   : security_config.py
#   Description : config file for symmetric encryption and decryption
#
#================================================================
import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

PASSWORD = b"password"
SALT = os.urandom(16)

def getKDF(shared_salt):
    return PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=shared_salt, iterations=100000, backend=default_backend())
def getKey(kdf_obj):
    return base64.urlsafe_b64encode(kdf_obj.derive(PASSWORD))
def getFernet(key):
    return Fernet(key)
def encrypt(fernet, plainText):
    return fernet.encrypt(plainText)
def decrypt(fernet, cipherText):
    return fernet.decrypt(cipherText)