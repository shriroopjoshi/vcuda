#include <iostream>
#include "vcuda.h"

#define N 64

using namespace std;

int main() {
    int a[N];
    int b[N];
    int c[N];
    label_t labels[4];
    for(int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = -i;
        c[i] = 1;
    }
    int n = N;
    vcuda_client client ("localhost", 9000);
    labels[0] = client.vcudaMalloc(1, VC_INT);
    labels[1] = client.vcudaMalloc(N, VC_INT);
    labels[2] = client.vcudaMalloc(N, VC_INT);
    labels[3] = client.vcudaMalloc(N, VC_INT);
    client.vcudaMemcpy(labels[0], &n, 1, vcudaMemcpyHostToDevice);
    client.vcudaMemcpy(labels[1], c, N, vcudaMemcpyHostToDevice);
    client.vcudaMemcpy(labels[2], a, N, vcudaMemcpyHostToDevice);
    client.vcudaMemcpy(labels[3], b, N, vcudaMemcpyHostToDevice);
    label_t kr = client.add_kernel("add.kr", "add");
    vcuda_dim3 bl (1);
    vcuda_dim3 th (N);
    client.execute_kernel(kr, bl, th, labels, 3);
    return 0;
}