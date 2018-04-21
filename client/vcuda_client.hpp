#ifndef VCUDA_CLIENT_H_
#define VCUDA_CLIENT_H_

#include "include/rapidjson/document.h"

using namespace rapidjson;

class vcuda_client {
    Document cuda_document;
    
    void print_document();
public:
    vcuda_client();
};

#endif