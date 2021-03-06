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
    return re.sub("\x00", "", data)

def sendall(sock, data):
    resp_str = str(data).encode("ascii", "replace")
    l = format(len(resp_str), "05")
    sock.send(l)
    sock.send(resp_str)

def writeprogramfile(filename, data, kernelfile):
    headers = ['#include "../vcuda_header.h"\n', '#include "' + kernelfile + '"\n',
    'using namespace std;\n', 'int main() {\n']
    vars = data["vars"]
    kernels = data["kernels"]
    with open(filename, "w") as sourcefile:
        sourcefile.writelines(headers)
        counter = {}
        c = 0
        for v in vars:
            counter[str(v)] = c
            c += 1
        # allocate host memory
        for v in vars:
            firsttime = True
            if v["type"] == "VC_INT":
                sourcefile.write('int h' + str(counter[str(v)]) + '[] = {')
            elif v["type"] == "VC_FLOAT":
                sourcefile.write('float h' + str(counter[str(v)]) + '[] = {')
            for i in v["data"]:
                if firsttime:
                    sourcefile.write(str(i))
                    firsttime = False
                else:
                    sourcefile.write(',' + str(i))
            sourcefile.write('};\n')
        # allocate device memory and initialize it
        for v in vars:
            if v["type"] == "VC_INT":
                sourcefile.write('int *d' + str(counter[str(v)]) + ';\n')
                sourcefile.write('cudaMalloc((void**)&d' + str(counter[str(v)]) + ',' + str(v["size"])  + '*sizeof(int));\n')
                sourcefile.write('cudaMemcpy(d' + str(counter[str(v)]) + ',h' + str(counter[str(v)]) + ', ' + str(v["size"]) + '*sizeof(int), cudaMemcpyHostToDevice);\n')
            elif v["type"] == "VC_FLOAT":
                sourcefile.write('float *d' + str(counter[str(v)]) + ';\n')
                sourcefile.write('cudaMalloc((void**)&d' + str(counter[str(v)]) + ',' + str(v["size"])  + '*sizeof(float));\n')
                sourcefile.write('cudaMemcpy(d' + str(counter[str(v)]) + ',h' + str(counter[str(v)]) + ', ' + str(v["size"]) + '*sizeof(float), cudaMemcpyHostToDevice);\n')
        kvars = 0
        for kernel in kernels:
            # allocate kernel blocks and threads
            if len(kernel["blocks"]) == 1:
                sourcefile.write('dim3 b' + str(kvars) + ' (' + str(kernel["blocks"][0]) + ');\n')
            elif len(kernel["blocks"]) == 2:
                sourcefile.write('dim3 b' + str(kvars) + ' (' + str(kernel["blocks"][0]) + ',' + str(kernel["blocks"][1]) + ');\n')
            else:
                sourcefile.write('dim3 b' + str(kvars) + ' (' + str(kernel["blocks"][0]) + ',' + str(kernel["blocks"][1]) + ',' + str(kernel["blocks"][2]) + ');\n')
            if len(kernel["threads"]) == 1:
                sourcefile.write('dim3 t' + str(kvars) + ' (' + str(kernel["threads"][0]) + ');\n')
            elif len(kernel["blocks"]) == 2:
                sourcefile.write('dim3 t' + str(kvars) + ' (' + str(kernel["threads"][0]) + ',' + str(kernel["threads"][1]) + ');\n')
            else:
                sourcefile.write('dim3 t' + str(kvars) + ' (' + str(kernel["threads"][0]) + ',' + str(kernel["threads"][1]) + ',' + str(kernel["threads"][2]) + ');\n')
            kvars += 1
            sourcefile.write(kernel["name"].split('.')[0] + '<<<b0,t0>>>(')
            index = 1
            firsttime = True
            for v in vars:
                if firsttime:
                    sourcefile.write('d' + str(counter[str(v)]))
                    firsttime = False
                else:
                    sourcefile.write(',d' + str(counter[str(v)]))
            sourcefile.write(');\n')
        for v in vars:
            if v["type"] == "VC_INT":
                sourcefile.write('cudaMemcpy(h' + str(counter[str(v)]) + ',d' + str(counter[str(v)]) + ', ' + str(v["size"]) + '*sizeof(int), cudaMemcpyDeviceToHost);\n')
            elif v["type"] == "VC_FLOAT":
                sourcefile.write('cudaMemcpy(h' + str(counter[str(v)]) + ',h' + str(counter[str(v)]) + ', ' + str(v["size"]) + '*sizeof(float), cudaMemcpyDeviceToHost);\n')
        for v in vars:
            sourcefile.write('bool first' + str(counter[str(v)]) + '=true;\n')
            sourcefile.write('for(int i=0;i<' + str(v["size"]) + ';i++){\n')
            sourcefile.write('if(first' + str(counter[str(v)]) + '){first' + str(counter[str(v)]) + '=false;cout<<h' + str(counter[str(v)]) + '[i];}\n')
            sourcefile.write('else{cout<<","<<h' + str(counter[str(v)]) + '[i];}\n}\ncout<<endl;\n')
        # free memory
        for v in vars:
            sourcefile.write('cudaFree(d' + str(counter[str(v)]) + ');\n')
        # end file
        sourcefile.write('return 0;\n}')



def clientthread(socket, address, id):
    print "client {" + str(id) + "} connected from " + address[0] + ":" + str(address[1])
    if os.path.exists(str(id)):
        shutil.rmtree(str(id))
    os.mkdir(str(id))
    data = recvall(socket)
    jsdata = json.loads(data.encode('utf-8').strip())
    kernelfile = ""
    if(jsdata.has_key("kernel")):
        kernelfile = jsdata["kernel"]["name"]
        filename = str(id) + "/" + kernelfile
        with open(filename, "w") as krfile:
            krfile.write(jsdata["kernel"]["code"])
        p = subprocess.Popen(["nvcc", "-x", "cu", "-c", filename, "-o", filename + ".o"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out, err) = p.communicate(None)
        print "OUTPUT:" + out
        print "ERROR:" + err
        resp = {}
        resp["output"] = out
        if err != "":
            resp["error"] = err
        else:
            resp["error"] = "none\n"
        if "error" in err:
            resp["stop"] = True
        else:
            resp["stop"] = False
        sendall(socket, json.dumps(resp))
        if "error" in err:
            return
    print "recieving"
    data = recvall(socket)
    print "done"
    print "data: \"" + data + "\""
    jsdata = json.loads(data.encode("utf-8").strip())
    filename = str(id) + "/client.cpp"
    writeprogramfile(filename, jsdata, kernelfile)
    p = subprocess.Popen(["nvcc", "-x", "cu", filename, "-o", str(id) + "/main"],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out, err) = p.communicate(None)
    print "OUTPUT: " + out
    print "ERROR: " + err
    resp = {}
    resp["output"] = out
    if err != "":
        resp["error"] = err
    else:
        resp["error"] = "none\n"
    if "error" in err:
        resp["stop"] = True
    else:
        resp["stop"] = False
    sendall(socket, json.dumps(resp))
    if "error" in err:
        # For next io.recv() call, send somthing.
        # This way client will not crash
        jsexit = {"exit": True}
        sendall(socket, json.dumps(jsexit))
        return
    p = subprocess.Popen([str(id) + "/main.exe"],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out, err) = p.communicate(None)
    print "EXE_OUT: " + out
    print "EXE_ERR: " + err
    outs = out.split('\n')
    i = 0
    for v in jsdata["vars"]:
        if i < len(jsdata["vars"]):
            resstring = outs[i].split(',')
            reslist = []
            if v["type"] == "VC_INT":
                for res in resstring:
                    reslist.append(int(res))
            elif v["type"] == "VC_FLOAT":
                for res in resstring:
                    reslist.append(float(res))
            v["data"] = reslist
            i += 1
    jsdata["errors"] = err
    sendall(socket, json.dumps(jsdata))
    
        

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
        # (clientsocket, address) = serversocket.accept()
        # id = uuid.uuid4()
        # clientthread(clientsocket, address, id)
    except KeyboardInterrupt:
        print "interrupted! Ctrl+C"
        serversocket.close()

start()


