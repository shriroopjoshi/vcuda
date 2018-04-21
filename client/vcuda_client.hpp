#ifndef VCUDA_CLIENT_H_
#define VCUDA_CLIENT_H_

#include "include/rapidjson/document.h"
#include "include/rapidjson/writer.h"
#include "include/rapidjson/stringbuffer.h"
#include <unordered_map>

using namespace rapidjson;

typedef int label_t;

enum vcuda_type {
    VC_INT, VC_FLOAT
};

enum vcuda_memcpy {
    vcudaMemcpyHostToDevice, vcudaMemcpyDeviceToHost
};

struct vcuda_var {
    int size;
    vcuda_type type;
};

class vcuda_client {
    Document cuda_document;
    std :: unordered_map<label_t, vcuda_var> inputs;
    void print_document();
public:
    vcuda_client();
    label_t vcudaMalloc(int, vcuda_type);
    void vcudaMemcpy(label_t, void *, int,  vcuda_memcpy);
};

#endif