#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <math.h>

#define Xs  10    
#define Ys  10
#define Zs  10
#define xs  10
#define ys  10
#define zs  10

#define VAR_T      0
#define VAR_MSG    1

#define TOTAL_E ((long long)Xs * Ys * Zs * xs * ys * zs)
#define UIDS TOTAL_E
#define CYCS_TO_RUN 25

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error " << cudaGetErrorString(err__)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

__host__ __device__ __forceinline__
long long UID_6D(int X, int Y, int Z, int x, int y, int z)
{
    const int outerX = Xs, outerY = Ys, outerZ = Zs;
    const int innerX = xs, innerY = ys, innerZ = zs;

    long long outerIndex = ((long long)Z * outerY + Y) * outerX + X;  // [Z][Y][X]
    long long innerIndex = ((long long)z * innerY + y) * innerX + x;  // [z][y][x]
    return outerIndex * ((long long)innerX * innerY * innerZ) + innerIndex;
}

__device__
double reconstructTemporalValueForOrder(
    const double* E,
    int k0,
    int Y, int Z, int x, int y, int z,
    double h,
    int max_order_use
){
    if (k0 < 0 || k0 >= Xs || Y < 0 || Y >= Ys || Z < 0 || Z >= Zs)
        return 0.0;

    int last_order = max_order_use;
    if (last_order < 0 || last_order >= Xs)
        last_order = Xs - 1;

    int N_terms = last_order - k0 + 1;
    if (N_terms <= 0) {
        long long uid = UID_6D(k0, Y, Z, x, y, z);
        return E[uid];
    }

    double value_out = 0.0;
    double power     = 1.0;   // h^0
    double factorial = 1.0;   // 0!

    for (int m = 0; m < N_terms; ++m) {
        int k = k0 + m;
        long long uid = UID_6D(k, Y, Z, x, y, z);
        double deriv_k = E[uid];

        if (m > 0) {
            power    *= h;          // h^m
            factorial *= (double)m; // m!
        }

        double coeff = power / factorial;
        value_out += coeff * deriv_k;
    }

    return value_out;
}

__global__
void persistentKernelSingleBlock(double* E,
                                 volatile int* d_CYC,
                                 volatile int* d_quit,
                                 volatile int* d_done)
{
    __shared__ int lastCYC;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        lastCYC = -1;
    __syncthreads();

    while (true) {
        int quit   = *d_quit;
        int curCYC = *d_CYC;

        if (quit) return;
        if (curCYC == lastCYC) continue;

        long long CYC = (long long)curCYC;
        double dt = 1.0;
        int z_cur = (int)(CYC % zs);

        // ---- Phase 1: update VAR_T / VAR_MSG over "logical" Y,Z,x,y,z ----

        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int tz = threadIdx.z;

        // Example mapping: each thread handles a slice of (Y,Z,x,y,z)
        // so we still get parallelism inside the block.
        for (int Y = 0; Y < Ys; ++Y) {
            for (int Z = 0; Z < Zs; ++Z) {
                // stride x,y,z by threadIdx
                for (int x = tx; x < xs; x += blockDim.x) {
                    for (int y = ty; y < ys; y += blockDim.y) {
                        int z = z_cur; // only current history slice

                        if (Y == VAR_T) {
                            double t = (double)CYC;
                            E[UID_6D(0, Y, Z, x, y, z)] = t;
                            E[UID_6D(1, Y, Z, x, y, z)] = 1.0;
                            for (int k = 2; k < Xs; ++k) {
                                E[UID_6D(k, Y, Z, x, y, z)] = 0.0;
                            }
                        } else if (Y == VAR_MSG) {
                            double f0 = E[UID_6D(0, Y, Z, x, y, z)];
                            double d1 = 0.0;
                            double d2 = 0.0;
                            if (CYC == 0) {
                                d1 = d2 = 0.0;
                            } else if (CYC == 1) {
                                int z_1 = (z_cur - 1 + zs) % zs;
                                double f1 = E[UID_6D(0, Y, Z, x, y, z_1)];
                                d1 = (f0 - f1) / dt;
                                d2 = 0.0;
                            } else {
                                int z_1 = (z_cur - 1 + zs) % zs;
                                int z_2 = (z_cur - 2 + zs) % zs;
                                double f1 = E[UID_6D(0, Y, Z, x, y, z_1)];
                                double f2 = E[UID_6D(0, Y, Z, x, y, z_2)];
                                d1 = (3.0 * f0 - 4.0 * f1 + f2) / (2.0 * dt);
                                d2 = (f0 - 2.0 * f1 + f2) / (dt * dt);
                            }
                            E[UID_6D(1, Y, Z, x, y, z)] = d1;
                            E[UID_6D(2, Y, Z, x, y, z)] = d2;
                        }
                    }
                }
                __syncthreads();
            }
        }

        // ---- Phase 2: debug print by one thread ----
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            int msg_len = zs;
            if (CYC >= msg_len - 1) {
                int Z0 = 0, x0 = 0, y0 = 0;
                double t_back = reconstructTemporalValueForOrder(
                    E, 0, VAR_T, Z0, x0, y0, z_cur,
                    (2.0 - (double)CYC),
                    Xs - 1
                );
                printf("CYC=%2lld  t_cur=%4.1f  t_back=%4.1f  msg_back=",
                       CYC, (double)CYC, t_back);
                for (int j = 0; j < msg_len; ++j) {
                    int z_j = j % zs;
                    char c = (char)(int)lrint(
                        E[UID_6D(0, VAR_MSG, Z0, x0, y0, z_j)]
                    );
                    if (c == 0) c = '_';
                    printf("%c", c);
                }
                printf("\n");
            }
            lastCYC = curCYC;
            *d_done = curCYC;
        }

        __syncthreads();
    }
}



int main()
{
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    int *h_done = nullptr;
    volatile int *d_done = nullptr;

    CUDA_CHECK(cudaHostAlloc(&h_done, sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((int**)&d_done, h_done, 0));
    *h_done = -1;

    double *h_E = nullptr;
    double *d_E = nullptr;
    int *h_CYC = nullptr, *h_quit = nullptr;
    volatile int *d_CYC = nullptr, *d_quit = nullptr;

    // Allocate pinned, mapped host memory for E
    CUDA_CHECK(cudaHostAlloc(&h_E, UIDS * sizeof(double), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_E, h_E, 0));

    // Allocate pinned, mapped host memory for control variables
    CUDA_CHECK(cudaHostAlloc(&h_CYC, sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&h_quit,  sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((int**)&d_CYC, h_CYC, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((int**)&d_quit,  h_quit,  0));

    for (int i = 0; i < UIDS; ++i) {
        h_E[i] = 0.0;
    }
    *h_CYC = 0;
    *h_quit  = 0;

    dim3 blocks(1, 1, 1);
    dim3 threads(xs, ys, zs);

    persistentKernelSingleBlock<<<blocks, threads>>>(d_E, (int*)d_CYC, (int*)d_quit, (int*)d_done);
    CUDA_CHECK(cudaGetLastError());

    // Host updates over CYCs
    for (int CYC = 0; CYC <= CYCS_TO_RUN; ++CYC) {

        // inject "hello world" into h_E
        for (int Z = 0; Z < Zs; ++Z) {
            for (int x = 0; x < xs; ++x) {
                for (int y = 0; y < ys; ++y) {
                    char c = 0;
                    if      (CYC == 0) c = 'h';
                    else if (CYC == 1) c = 'e';
                    else if (CYC == 2) c = 'l';
                    else if (CYC == 3) c = 'l';
                    else if (CYC == 4) c = 'o';
                    else if (CYC == 5) c = 'w';
                    else if (CYC == 6) c = 'o';
                    else if (CYC == 7) c = 'r';
                    else if (CYC == 8) c = 'l';
                    else if (CYC == 9) c = 'd';
                    else c = 0;

                    if (c != 0) {
                        h_E[UID_6D(0,VAR_MSG,Z,x,y,(CYC%zs))] = (double)c;
                    }
                }
            }
        }

        *h_CYC = CYC;

        // Wait until GPU acknowledges this CYC
        while (*h_done < CYC) {
            // optional: tiny sleep
            // std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    *h_quit = 1;

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFreeHost(h_E));
    CUDA_CHECK(cudaFreeHost(h_CYC));
    CUDA_CHECK(cudaFreeHost(h_quit));
    CUDA_CHECK(cudaFreeHost(h_done));

    return 0;
}
