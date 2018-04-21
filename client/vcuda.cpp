#include <iostream>
#include "vcuda_client.hpp"

vcuda_client client;

int main() {
    label_t c1 = client.vcudaMalloc(3, VC_INT);
    std :: cout << "c = " << c1 << std :: endl;
    label_t c2 = client.vcudaMalloc(5, VC_FLOAT);
    std :: cout << "c = " << c2 << std :: endl;    
    int arr[] = {1, 2, 3};
    client.vcudaMemcpy(c1, arr, 3, vcudaMemcpyDeviceToHost);
    float ptr[] = {2.5, 7.9, 6.3, 9.45, 8.2385};
    client.vcudaMemcpy(c2, ptr, 5, vcudaMemcpyDeviceToHost);
    return 0;
}