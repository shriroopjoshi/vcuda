#include "vcuda_io.h"

void vcuda_io :: init(std :: string host, int port) {
    hostent *server;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if(sockfd < 0) {
        sockfd = -1;
    } else {
        server = gethostbyname(host.c_str());
        if(server == NULL) {
            sockfd = -2;
        } else {
            bzero((char *) &serv_addr, sizeof(serv_addr));
            serv_addr.sin_family = AF_INET;
            bcopy((char *) server -> h_addr, (char *) &serv_addr.sin_addr.s_addr, server -> h_length);
            serv_addr.sin_port = htons(port);
        }
    }
}

int vcuda_io :: connect_server() {
    if(sockfd == -1 || sockfd == -2) {
        return sockfd;
    }
    if(connect(sockfd, (sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        return -4;
    } else {
        return 0;
    }
}

int vcuda_io :: send(std :: string str) {
    char buffer[BUFFER_SIZE];
    bzero(buffer, BUFFER_SIZE);
    size_t size = str.size();
    std :: string substring;
    int n = 0, i = 0;
    while(size > BUFFER_SIZE) {
        substring = str.substr(i * BUFFER_SIZE, (i + 1) * BUFFER_SIZE - 1);
        strcpy(buffer, substring.c_str());
        n += write(sockfd, buffer, BUFFER_SIZE);
        size = size - BUFFER_SIZE;
        i += 1;
    }
    substring = str.substr(i * BUFFER_SIZE, size).c_str();
    strcpy(buffer, substring.c_str());
    n += write(sockfd, buffer, strlen(buffer));
    return n;
}

std :: string vcuda_io :: read_kernel(std :: string filename) {
    std :: ifstream in (filename.c_str(), std :: ios :: in | std :: ios :: binary);
    std :: string code;
    if(in) {
        in.seekg(0, std::ios::end);
        code.resize(in.tellg());
        in.seekg(0, std::ios::beg);
        in.read(&code[0], code.size());
        in.close();
        return code;
    } else {
        return "err";
    }
}

void vcuda_io :: disconnect() {
    close(sockfd);
}