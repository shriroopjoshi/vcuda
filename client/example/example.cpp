#include "vcuda.h"
#include <iostream>

int main() {
    vcuda_client client ("localhost", 9000);
    label_t cc[2];
    cc[0] = client.vcudaMalloc(3, VC_INT);
    std :: cout << "c = " << cc[0] << std :: endl;
    cc[1] = client.vcudaMalloc(5, VC_FLOAT);
    std :: cout << "c = " << cc[1] << std :: endl;    
    int arr[] = {1, 2, 3};
    client.vcudaMemcpy(cc[0], arr, 3, vcudaMemcpyDeviceToHost);
    float ptr[] = {2.5, 7.9, 6.3, 9.45, 8.2385};
    client.vcudaMemcpy(cc[1], ptr, 5, vcudaMemcpyDeviceToHost);
    label_t k1 = client.add_kernel("example.kr", "saxpy");
    vcuda_dim3 b (1);
    vcuda_dim3 t (10);
    client.execute_kernel(k1, b, t, cc, 2);
    std :: cout << "graceful ending!" << std :: endl;
    return 0;
}