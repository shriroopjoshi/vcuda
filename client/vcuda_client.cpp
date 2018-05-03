#include "vcuda.h"

using namespace rapidjson;

vcuda_client :: vcuda_client(std :: string host, int port) {
    io.init(host, port);
    const char *init_str = "{\"vars\": [], \"kernels\": []}";
    cuda_document.Parse(init_str);
}

vcuda_client :: ~vcuda_client() {
    io.disconnect();
}

label_t vcuda_client :: vcudaMalloc(int n, vcuda_type type) {
    label_t label = (label_t) inputs.size() + 1;
    vcuda_var variable;
    variable.label = label;
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
    v.AddMember("label", variable.label, allocator);
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
    if(kernel_str == "err") {
        std :: cerr << name << " - Kernel file not found!" << std :: endl;
        exit(1);
    }
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
    v.AddMember("code", kernel_val, allocator);
    kdoc.AddMember("kernel", v, allocator);
    std :: pair <label_t, std :: string> p (label, name);
    kernels.insert(p);
    err_exit(io.connect_server());
    io.send(print_document(kdoc));
    delete[] kr_str;
    std :: string recv_data = io.recv();
    Document resp;
    resp.Parse(recv_data.c_str());
    if(strlen(resp["output"].GetString()) > 0) {
        std :: cout << "output: " <<  resp["output"].GetString();
    }
    if(strlen(resp["error"].GetString()) > 0) {
        std :: cerr << "error: " <<  resp["error"].GetString();
    }
    if(resp["stop"].GetBool()) {
        exit(1);
    }
    return label;
}

void vcuda_client :: execute_kernel(label_t kernel_label, vcuda_dim3 blocks, vcuda_dim3 threads, label_t params[], int n) {
    err_exit(io.connect_server());
    // io.send(print_document(cuda_document));
    Document::AllocatorType& allocator = cuda_document.GetAllocator();
    Value& kr = cuda_document["kernels"];
    Value v (kObjectType);
    Value bdim (kArrayType);
    Value tdim (kArrayType);
    bdim.PushBack(blocks.x, allocator).PushBack(blocks.y, allocator).PushBack(blocks.z, allocator);
    tdim.PushBack(threads.x, allocator).PushBack(threads.y, allocator).PushBack(threads.z,allocator);
    v.AddMember("blocks", bdim, allocator);
    v.AddMember("threads", tdim, allocator);
    std :: unordered_map<label_t, std :: string> :: iterator it = kernels.find(kernel_label);
    if(it == kernels.end()) {
        std :: cerr << "No such kernel.\nAre you sure kernel_label is correct?" << std :: endl;
        exit(1);
    } else {
        std :: string s = it -> second;
        char nameb[s.size() + 1];
        strcpy(nameb, s.c_str());
        Value name_value;
        name_value.SetString(nameb, s.length(), allocator);
        v.AddMember("name", name_value, allocator);
    }
    Value ins (kArrayType);
    for(int i = 0; i < n; ++i) {
        ins.PushBack(params[i], allocator);
    }
    v.AddMember("params", ins, allocator);
    kr.PushBack(v, allocator);
    io.send(print_document(cuda_document));
    std :: cout << io.recv() << std :: endl;
    std :: cout << io.recv() << std :: endl;
}

std :: string vcuda_client :: print_document(Document &d) {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    d.Accept(writer);
    std :: cout << buffer.GetString() << std :: endl;
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