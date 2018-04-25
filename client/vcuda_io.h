#ifndef VCUDA_IO_H_
#define VCUDA_IO_H_

#include <unistd.h>
#include <fstream>
#include <cstring>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 

#define BUFFER_SIZE 1024

class vcuda_io {
    sockaddr_in serv_addr;
    int sockfd;
public:
    void init(std :: string, int);
    int connect_server();
    int send(std :: string);
    std :: string read_kernel(std :: string);
    void disconnect();
};

#endif