#include "vcuda_client.hpp"
#include "include/rapidjson/document.h"
#include "include/rapidjson/writer.h"
#include "include/rapidjson/stringbuffer.h"
#include <iostream>

using namespace rapidjson;

vcuda_client :: vcuda_client() {
    const char *init_str = "{\"vars\": {}, \"result\": {}, \"kernel\": {}}";
    cuda_document.Parse(init_str);
    print_document();
}

void vcuda_client :: print_document() {
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    cuda_document.Accept(writer);
    std :: cout << buffer.GetString() << std :: endl;
}