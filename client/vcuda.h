#ifndef VCUDA_H_
#define VCUDA_H_

#include "include/rapidjson/document.h"
#include "include/rapidjson/writer.h"
#include "include/rapidjson/stringbuffer.h"
#include "vcuda_io.h"

#include <iostream>
#include <unordered_map>
#include <cstdlib>
#include <unistd.h>
#include <fstream>
#include <streambuf>
#include <cerrno>
#include <string>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 

using namespace rapidjson;

#define BUFFER_SIZE 1024

typedef int label_t;

enum vcuda_type {
    VC_INT, VC_FLOAT
};

enum vcuda_memcpy {
    vcudaMemcpyHostToDevice, vcudaMemcpyDeviceToHost
};

struct vcuda_dim3 {
    int x, y, z;

    vcuda_dim3(int x) {
        this -> x = x;
        y = 1;
        z = 1;
    }

    vcuda_dim3(int x, int y) {
        this -> x = x;
        this -> y = y;
        z = 1;
    }

    vcuda_dim3(int x, int y, int z) {
        this -> x = x;
        this -> y = y;
        this -> z = z;
    }
};

struct vcuda_var {
    label_t label;
    int size;
    vcuda_type type;
};

class vcuda_client {
    Document cuda_document;
    vcuda_io io;
    std :: unordered_map<label_t, vcuda_var> inputs;
    std :: unordered_map<label_t, std :: string> kernels;
    std :: string print_document(Document &);
    void vcudaMemcpyToDevice(label_t, void *, int);
    void vcudaMemcpyToHost(label_t, void *, int);
    void err_exit(int);
public:
    vcuda_client(std :: string, int);
    ~vcuda_client();
    label_t add_kernel(std :: string, std :: string);
    void set_kernel_parameters(label_t, vcuda_dim3, vcuda_dim3);
    label_t vcudaMalloc(int, vcuda_type);
    void vcudaMemcpy(label_t, void *, int,  vcuda_memcpy);
    void execute_kernel(label_t, vcuda_dim3, vcuda_dim3, label_t *, int);
};

#endif