#include "vcuda.h"
#include <iostream>

using namespace rapidjson;

vcuda_client :: vcuda_client(std :: string host, int port) {
    this -> host = host;
    this -> port = port;
    const char *init_str = "{\"vars\": [], \"result\": {}, \"kernel\": []}";
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
    print_document();
}

label_t vcuda_client :: add_kernel(std :: string path) {
    label_t label = (label_t) kernels.size() + 1;
    std :: ifstream in(path.c_str(), std :: ios :: in | std :: ios :: binary);
    std :: string kernel_str;
    if(in) {
        in.seekg(0, std::ios::end);
        kernel_str.resize(in.tellg());
        in.seekg(0, std::ios::beg);
        in.read(&kernel_str[0], kernel_str.size());
        in.close();
    } else {
        throw (errno);
    }
    Document :: AllocatorType& allocator = cuda_document.GetAllocator();
    Value& doc_kernels = cuda_document["kernel"];
    Value v (kObjectType);
    v.AddMember("blocks", 1, allocator);
    v.AddMember("threads", 1, allocator);
    Value kernel_val;
    char *kr_str = new char[kernel_str.length() + 1];
    strcpy(kr_str, kernel_str.c_str());
    kernel_val.SetString(kr_str, kernel_str.length() + 1, allocator);
    v.AddMember("code", kernel_val, allocator);
    doc_kernels.PushBack(v, allocator);
    std :: pair <label_t, std :: string> p (label, path);
    kernels.insert(p);
    delete[] kr_str;
    return label;
}

void vcuda_client :: execute_kernel(label_t kernel_label) {
    int sockfd, n;
    sockaddr_in serv_addr;
    hostent *server;

    char buffer[BUFFER_SIZE];
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if(sockfd < 0) {
        std :: cerr << "Unable to open socket" << std :: endl;
        return;
    }
    server = gethostbyname(host.c_str());
    if(server == NULL) {
        std :: cerr << "No such server" << std :: endl;
        return;
    }
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *) server -> h_addr, (char *) &serv_addr.sin_addr.s_addr, server -> h_length);
    serv_addr.sin_port = htons(port);
    if(connect(sockfd, (sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        std :: cerr << "Error connecting" << std :: endl;
        return;
    }
    bzero(buffer, BUFFER_SIZE);
    std :: string json = print_document();
    size_t size = json.size();
    std :: string substring;
    int i = 0;
    while(size > BUFFER_SIZE) {
        substring = json.substr(i * BUFFER_SIZE, (i + 1) * BUFFER_SIZE - 1);
        strcpy(buffer, substring.c_str());
        n = write(sockfd, buffer, BUFFER_SIZE);
        size = size - BUFFER_SIZE;
        i += 1;
    }
    substring = json.substr(i * BUFFER_SIZE, size).c_str();
    strcpy(buffer, substring.c_str());
    n = write(sockfd, buffer, strlen(buffer));
    close(sockfd);
}

std :: string vcuda_client :: print_document() {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    cuda_document.Accept(writer);
    // std :: cout << buffer.GetString() << std :: endl;
    return buffer.GetString();
}