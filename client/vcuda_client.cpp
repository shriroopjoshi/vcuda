#include "vcuda_client.hpp"
#include <iostream>

using namespace rapidjson;

vcuda_client :: vcuda_client() {
    const char *init_str = "{\"vars\": [], \"result\": {}, \"kernel\": {}}";
    cuda_document.Parse(init_str);
    print_document();
}

label_t vcuda_client :: vcudaMalloc(int n, vcuda_type type) {
    label_t label = (label_t) inputs.size() + 1;
    vcuda_var variable;
    variable.size = n;
    variable.type = type;
    std :: pair<label_t, vcuda_var> p(label, variable);
    inputs.insert(p);
    return label;
}

void vcuda_client :: vcudaMemcpy(label_t label, void *ptr, int count, vcuda_memcpy kind) {
    vcuda_var variable = inputs[label];
    Document :: AllocatorType& allocator = cuda_document.GetAllocator();
    Value& doc_vars = cuda_document["vars"];
    Value v(kObjectType);
    v.AddMember("size", variable.size, allocator);
    if(variable.type == VC_INT) {
        v.AddMember("type", "VC_INT", allocator);
        Value a (kArrayType);
        for(int i = 0; i < count; ++i) {
            a.PushBack(((int*) ptr)[i], allocator);
        }
        v.AddMember("data", a, allocator);
    } else if(variable.type == VC_FLOAT) {
        v.AddMember("type", "VC_FLOAT", allocator);
        Value a (kArrayType);
        for(int i = 0; i < count; ++i) {
            a.PushBack(((float*) ptr)[i], allocator);
        }
        v.AddMember("data", a, allocator);
    }
    doc_vars.PushBack(v, allocator);
    print_document();
}

void vcuda_client :: print_document() {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    cuda_document.Accept(writer);
    std :: cout << buffer.GetString() << std :: endl;
}