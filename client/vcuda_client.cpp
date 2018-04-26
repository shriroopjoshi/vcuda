#include "vcuda.h"

using namespace rapidjson;

vcuda_client :: vcuda_client(std :: string host, int port) {
    io.init(host, port);
    const char *init_str = "{\"vars\": []}";
    cuda_document.Parse(init_str);
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
            a.PushBack(((int *) ptr)[i], allocator);
        }
        v.AddMember("data", a, allocator);
    } else if(variable.type == VC_FLOAT) {
        v.AddMember("type", "VC_FLOAT", allocator);
        Value a (kArrayType);
        for(int i = 0; i < count; ++i) {
            a.PushBack(((float *) ptr)[i], allocator);
        }
        v.AddMember("data", a, allocator);
    }
    doc_vars.PushBack(v, allocator);
}

label_t vcuda_client :: add_kernel(std :: string path, std :: string name) {
    label_t label = (label_t) kernels.size() + 1;
    std :: string kernel_str;
    kernel_str = io.read_kernel(path);
    char *kr_str = new char[kernel_str.length() + 1];
    strcpy(kr_str, kernel_str.c_str());
    Document kdoc;
    kdoc.Parse("{}");
    Document :: AllocatorType& allocator = kdoc.GetAllocator();
    Value kernel_val;
    kernel_val.SetString(kr_str, kernel_str.length(), allocator);
    name.append(".kr");
    char *nameb = new char[name.length() + 1];
    strcpy(nameb, name.c_str());
    Value name_value;
    name_value.SetString(nameb, name.length(), allocator);
    Value v (kObjectType);
    v.AddMember("name", name_value, allocator);
    v.AddMember("blocks", 1, allocator);
    v.AddMember("threads", 1, allocator);
    v.AddMember("code", kernel_val, allocator);
    kdoc.AddMember("kernel", v, allocator);
    std :: pair <label_t, std :: string> p (label, path);
    kernels.insert(p);
    err_exit(io.connect_server());
    io.send(print_document(kdoc));
    delete[] kr_str;
    std :: string recv_data = io.recv();
    Document resp;
    resp.Parse(recv_data.c_str());
    std :: cout << resp["output"].GetString() << std :: endl;
    std :: cerr << resp["error"].GetString() << std :: endl;
    if(resp["stop"].GetBool()) {
        exit(1);
    }
    return label;
}

void vcuda_client :: execute_kernel(label_t kernel_label) {
    err_exit(io.connect_server());
    // io.send(print_document(cuda_document));
    io.disconnect();
}

std :: string vcuda_client :: print_document(Document &d) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    d.Accept(writer);
    // std :: cout << buffer.GetString() << std :: endl;
    return buffer.GetString();
}

void vcuda_client :: err_exit(int status) {
    if(status >= 0) {
        return;
    }
    if(status == -1) {
        std :: cerr << "Unable to open a connection to server" << std :: endl;
    }
    if(status == -2) {
        std :: cerr << "Unable to find server" << std :: endl;
    }
    if(status == -4) {
        std :: cerr << "Unable to connect to server" << std :: endl;
    }
    exit(status * -1);
}