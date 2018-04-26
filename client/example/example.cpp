#include "vcuda.h"
#include <iostream>

int main() {
    vcuda_client client ("localhost", 9000);
    label_t c1 = client.vcudaMalloc(3, VC_INT);
    std :: cout << "c = " << c1 << std :: endl;
    label_t c2 = client.vcudaMalloc(5, VC_FLOAT);
    std :: cout << "c = " << c2 << std :: endl;    
    int arr[] = {1, 2, 3};
    client.vcudaMemcpy(c1, arr, 3, vcudaMemcpyDeviceToHost);
    float ptr[] = {2.5, 7.9, 6.3, 9.45, 8.2385};
    client.vcudaMemcpy(c2, ptr, 5, vcudaMemcpyDeviceToHost);
    label_t k1 = client.add_kernel("example.kr", "example");
    client.execute_kernel(k1);
    std :: cout << "graceful ending!" << std :: endl;
    return 0;
}