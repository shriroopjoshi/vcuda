#! /bin/python

from socket import *
from socket import error as socket_error
from thread import *
import sys
import json
import re
import uuid
import os
import shutil
import subprocess

HOST = "localhost"
PORT = 9000

"""
Reads all contents from a socket and returns it
"""
def recvall(sock):
    BUFF_SIZE = 1024
    data = b''
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        if len(part) < BUFF_SIZE:
            # either 0 or eof
            break
    return data

def clientthread(socket, address, id):
    print "client {" + str(id) + "} connected from " + address[0] + ":" + str(address[1])
    if os.path.exists(str(id)):
        shutil.rmtree(str(id))
    os.mkdir(str(id))
    data = recvall(socket)
    data = re.sub("\x00", "", data)
    jsdata = json.loads(data.encode('utf-8').strip())
    if(jsdata.has_key("kernel")):
        filename = str(id) + "/" + jsdata["kernel"]["name"]
        with open(filename, "w") as krfile:
            krfile.write(jsdata["kernel"]["code"])
        p = subprocess.Popen(["nvcc", "-x", "cu", "-c", filename, "-o", filename + ".o"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out, err) = p.communicate(None)
        print "OUTPUT:" + out
        print "ERROR:" + err
        resp = {}
        resp["output"] = out
        resp["error"] = err
        if "error" in err:
            resp["stop"] = True
        else:
            resp["stop"] = False
        json_resp = json.dumps(resp)
        resp_str = str(json_resp).encode("ascii", "replace")
        l = format(len(resp_str), "05")
        print l
        socket.send(l)
        socket.send(resp_str)




def start():
    serversocket = socket(AF_INET, SOCK_STREAM)
    try:
        serversocket.bind((HOST, PORT))
    except socket_error as msg:
        print "Unable to start server"
        print "Error code: " + str(msg[0]) + " - " + msg[1]
        sys.exit()
    serversocket.listen(5)
    try:
        while True:
            (clientsocket, address) = serversocket.accept()
            id = uuid.uuid4()
            start_new_thread(clientthread, (clientsocket, address, id, ))
    except KeyboardInterrupt:
        print "interrupted! Ctrl+C"
        serversocket.close()

start()


