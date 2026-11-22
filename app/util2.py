import os
import numpy as np
import hashlib
from Crypto.Cipher import AES
from secrets import token_bytes, token_hex
from Crypto.Random import get_random_bytes
from dotenv import load_dotenv

load_dotenv()

key = bytes.fromhex(os.getenv("AES_SECRET_KEY"))



def encrypt_embedding(embedding):
    embedding_bytes = embedding.tobytes()
    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(embedding_bytes)
    # print(hashlib.sha256(key).hexdigest())

    return nonce + ciphertext + tag


def decrypt_embedding(data):
    nonce, ciphertext, tag = data[:12], data[12:-16], data[-16:]
   
    receiver = AES.new(key, AES.MODE_GCM, nonce=nonce)
    plaintext = receiver.decrypt_and_verify(ciphertext, tag)

    return plaintext


# def test_encryption(embedding):

#     embedding_bytes = embedding.tobytes()

#     nonce = get_random_bytes(12)

#     cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)

#     ciphertext, tag = cipher.encrypt_and_digest(embedding_bytes)

#     receiver = AES.new(key, AES.MODE_GCM, nonce=nonce)

#     plaintext = receiver.decrypt_and_verify(ciphertext, tag)

#     print("Original:", embedding_bytes == plaintext)

#     return plaintext