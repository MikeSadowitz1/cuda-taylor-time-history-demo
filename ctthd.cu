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

#define TOTAL_E  ((long long)Xs * Ys * Zs * xs * ys * zs)
#define UIDS     TOTAL_E
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

    long long outerIndex = ((long long)Z * outerY + Y) * outerX + X;
    long long innerIndex = ((long long)z * innerY + y) * innerX + x;
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
    double power     = 1.0;
    double factorial = 1.0;

    for (int m = 0; m < N_terms; ++m) {
        int k = k0 + m;
        long long uid = UID_6D(k, Y, Z, x, y, z);
        double deriv_k = E[uid];

        if (m > 0) {
            power    *= h;
            factorial *= (double)m;
        }

        double coeff = power / factorial;
        value_out += coeff * deriv_k;
    }

    return value_out;
}

// Per-cycle barrier state
// Per-cycle barrier state
__device__ volatile unsigned int g_threadsDone  = 0u;
__device__ volatile int          g_activeCycle  = -1;

__global__
void persistentKernel(double* E,volatile int* d_CYC,volatile int* d_quit,volatile int* d_done){
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int numThreads = gridDim.x * blockDim.x;

    const unsigned int TOTAL_TASKS = (unsigned int)Ys * (unsigned int)Zs * (unsigned int)xs * (unsigned int)ys;

    __syncthreads();

    int localLastCYC = -1;

    while (true) {

        if (*d_quit) return;

        int curCYC = *d_CYC;
        if (curCYC < 0) {
            continue;
        }
        if (curCYC == localLastCYC) {
            continue;
        }

        // One control thread sets up per-cycle state
        if (tid == 0) {
            g_activeCycle = curCYC;
            g_threadsDone = 0u;
            __threadfence();      // make these writes visible to all SMs
        }

        // Wait until the device-wide cycle value matches curCYC
        while (g_activeCycle != curCYC) {
            // optional: __nanosleep(100);
        }

        long long CYC = (long long)curCYC;
        double dt = 1.0;
        int z_cur = (int)(CYC % zs);

        // Each thread processes a strided subset of the logical grid
        for (unsigned int t = tid; t < TOTAL_TASKS; t += numThreads) {

            unsigned int tmp = t;

            int y = (int)(tmp % ys); tmp /= ys;
            int x = (int)(tmp % xs); tmp /= xs;
            int Z = (int)(tmp % Zs); tmp /= Zs;
            int Y = (int)(tmp % Ys);

            int z = z_cur;

            if (Y == VAR_T) {
                double tcur = (double)CYC;
                E[UID_6D(0, Y, Z, x, y, z)] = tcur;
                E[UID_6D(1, Y, Z, x, y, z)] = 1.0;
                for (int k = 2; k < Xs; ++k)
                    E[UID_6D(k, Y, Z, x, y, z)] = 0.0;

            } else if (Y == VAR_MSG) {
                double f0 = E[UID_6D(0, Y, Z, x, y, z)];
                double d1 = 0.0, d2 = 0.0;

                if (CYC == 0) {
                    d1 = d2 = 0.0;
                } else if (CYC == 1) {
                    int z_1 = (z_cur - 1 + zs) % zs;
                    double f1 = E[UID_6D(0, Y, Z, x, y, z_1)];
                    d1 = (f0 - f1) / dt;
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

        // Every active worker thread signals completion
        unsigned int finished = atomicAdd((unsigned int*)&g_threadsDone, 1u) + 1u;

        if (finished == numThreads) {
            int Z0 = 0, x0 = 0, y0 = 0;
            int msg_len = zs;

            if (CYC >= msg_len - 1) {
                
                printf("CYC=%2lld  t_cur=%4.1f  t_back=%4.1f  msg_back=",CYC, (double)CYC, reconstructTemporalValueForOrder(E,0,VAR_T,Z0,x0,y0,z_cur,(2.0 - (double)CYC),Xs - 1));

                for (int j = 0; j < msg_len; ++j) {
                    int z_j = j % zs;
                    char c = (char)(int)lrint(E[UID_6D(0,VAR_MSG,Z0,x0,y0,z_j)]);
                    if (c == 0) c = '_';
                    printf("%c", c);
                }
                printf("\n");
            }


            int x1 = 1;
            if (CYC >= msg_len - 1) {
                
                printf("CYC=%2lld  t_cur=%4.1f  t_back=%4.1f  msg_back=",CYC, (double)CYC, reconstructTemporalValueForOrder(E,0,VAR_T,Z0,x1,y0,z_cur,(2.0 - (double)CYC),Xs - 1));

                for (int j = 0; j < msg_len; ++j) {
                    int z_j = j % zs;
                    char c = (char)(int)lrint(E[UID_6D(0,VAR_MSG,Z0,x1,y0,z_j)]);
                    if (c == 0) c = '_';
                    printf("%c", c);
                }
                printf("\n");
            }




            *d_done = curCYC;
            __threadfence_system();  // flush to host-visible memory
        }

        localLastCYC = curCYC;
    }
}


int main()
{
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    // Figure out a safe number of blocks for a persistent kernel
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int blocksPerSM;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM,
        persistentKernel,
        256,  // threads per block (below)
        0     // dynamic shared memory
    ));

    int maxBlocks = blocksPerSM * prop.multiProcessorCount;
    // To be extra safe for persistence, we can cap at 1 block per SM:
    if (maxBlocks > prop.multiProcessorCount)
        maxBlocks = prop.multiProcessorCount;

    double *h_E = nullptr;
    double *d_E = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_E, UIDS * sizeof(double), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_E, h_E, 0));

    volatile int *h_CYC  = nullptr;
    volatile int *h_quit = nullptr;
    volatile int *h_done = nullptr;

    volatile int *d_CYC  = nullptr;
    volatile int *d_quit = nullptr;
    volatile int *d_done = nullptr;

    CUDA_CHECK(cudaHostAlloc((void**)&h_CYC,  sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void**)&h_quit, sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void**)&h_done, sizeof(int), cudaHostAllocMapped));

    CUDA_CHECK(cudaHostGetDevicePointer((int**)&d_CYC,  (void*)h_CYC,  0));
    CUDA_CHECK(cudaHostGetDevicePointer((int**)&d_quit, (void*)h_quit, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((int**)&d_done, (void*)h_done, 0));

    for (long long i = 0; i < UIDS; ++i) h_E[i] = 0.0;
    *h_CYC  = -1;
    *h_quit = 0;
    *h_done = -1;

    dim3 blocks(maxBlocks, 1, 1);
    dim3 threads(256, 1, 1);

    persistentKernel<<<blocks, threads>>>(d_E, d_CYC, d_quit, d_done);
    CUDA_CHECK(cudaGetLastError());

    const char* msg0 = "helloworld";
    const char* msg1 = "whatsgoing";

    for (int CYC = 0; CYC <= CYCS_TO_RUN; ++CYC) {

        if (CYC < 10) {
            char c0 = msg0[CYC];
            char c1 = msg1[CYC];
            int z_cur = CYC % zs;
            for (int Z = 0; Z < Zs; ++Z)
                for (int y = 0; y < ys; ++y)
                    for (int x = 0; x < xs; ++x)
                        if (x == 0){
                            h_E[UID_6D(0, VAR_MSG, Z, x, y, z_cur)] = (double)c0;
                        } else if (x == 1){
                            h_E[UID_6D(0, VAR_MSG, Z, x, y, z_cur)] = (double)c1;
                        }
        }

        *h_CYC = CYC;

        while (*h_done < CYC) {
            // optional small sleep here if you want
        }
    }

    *h_quit = 1;
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaFreeHost((void*)h_E));
    CUDA_CHECK(cudaFreeHost((void*)h_CYC));
    CUDA_CHECK(cudaFreeHost((void*)h_quit));
    CUDA_CHECK(cudaFreeHost((void*)h_done));

    return 0;
}

