// optionA_hybrid_demo.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <math.h>

// ---- Problem dimensions (small but fully 6D) ----
#define Xs  10    
#define Ys  10
#define Zs  10
#define xs  10
#define ys  10
#define zs  10

#define VAR_T      0
#define VAR_MSG    1

#define TOTAL_E ((long long)Xs * Ys * Zs * xs * ys * zs)

// ---- CUDA error checking ----
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

// ---------------------------------------------------------------------
// UID_6D: flatten (X,Y,Z,x,y,z) into [0, TOTAL_E)
//
// Layout:
//   Outer dims: [Z][Y][X]
//   Inner dims: [z][y][x]
//
// E[ UID_6D(X,Y,Z,x,y,z) ] = derivative order X for variable Y
//                           at spatial (Z,x,y) and history index z.
// ---------------------------------------------------------------------
__device__ __forceinline__
long long UID_6D(int X, int Y, int Z, int x, int y, int z)
{
    const int outerX = Xs, outerY = Ys, outerZ = Zs;
    const int innerX = xs, innerY = ys, innerZ = zs;

    long long outerIndex = ((long long)Z * outerY + Y) * outerX + X;  // [Z][Y][X]
    long long innerIndex = ((long long)z * innerY + y) * innerX + x;  // [z][y][x]
    return outerIndex * ((long long)innerX * innerY * innerZ) + innerIndex;
}

// ---------------------------------------------------------------------
// Taylor reconstruction for an ANALYTIC variable (e.g. time):
//
//   E^{(k0)}(t + h) ≈ Σ_{m=0}^{M} [ h^m / m! ] * E^{(k0+m)}(t)
//
// We will only use this for VAR_T (continuous time), not for VAR_MSG.
// ---------------------------------------------------------------------
__device__
double reconstructTemporalValueForOrder(
    const double* __restrict__ E,
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

// ---------------------------------------------------------------------
// 1) Inject discrete message characters into VAR_MSG history.
//
// We treat z_cur = CYC % zs as the "current history slot" on z.
// Here we hardcode msg(t):
//   CYC 0 -> 'h'
//   CYC 1 -> 'e'
//   CYC 2 -> 'l'
//   CYC 3 -> 'l'
//   CYC 4 -> 'o'
//  etc.
// IMPORTANT: this is a discrete sequence, not a smooth function.
// We will NOT use Taylor on VAR_MSG; just direct history lookup.
// ---------------------------------------------------------------------
__global__
void injectMsgHistoryKernel(long long CYC, double* __restrict__ E)
{
    int Y = VAR_MSG;
    int Z = blockIdx.z;
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;      // time-history index this thread "owns"

    int z_cur = (int)(CYC % zs);
    if (z != z_cur) return;   // only the current slice writes

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
    else               c = 0;  // no new character after (history remains)

    if (c != 0) {
        E[UID_6D(0, Y, Z, x, y, z)] = (double)c;
    }
}

// ---------------------------------------------------------------------
// 2) Fill analytic time variable VAR_T at z_cur.
//
// We define E_t(t) = t (continuous).
//  - E^0_t(t) = t
//  - E^1_t(t) = 1
//  - E^k_t(t) = 0 for k>=2
//
// These derivatives make Taylor reconstruction EXACT for time:
//   E_t(t_target) = t_target.
// ---------------------------------------------------------------------
__global__
void fillKsKernel(long long CYC, double dt,double* __restrict__ E)
{
    int Y = blockIdx.y;
    int Z = blockIdx.z;
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;

    int z_cur = (int)(CYC % zs);
    if (z != z_cur) return;   // only current history slice

    if (Y == VAR_T){
        double t = (double)CYC;
        E[UID_6D(0, Y, Z, x, y, z)] = t;    // value
        E[UID_6D(1, Y, Z, x, y, z)] = 1.0;  // first derivative
        for (int k = 2; k < Xs; ++k) {
            E[UID_6D(k, Y, Z, x, y, z)] = 0.0;
        }
    } else if (Y == VAR_MSG){
        double f0 = E[UID_6D(0, Y, Z, x, y, z)];

        double d1 = 0.0;
        double d2 = 0.0;

        if (CYC == 0) {
            d1 = 0.0;
            d2 = 0.0;
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

// ---------------------------------------------------------------------
// 3) Debug/inspect kernel:
//
//   - Uses Taylor on VAR_T to reconstruct t_target from t and d/dt.
//   - Uses direct z-history lookup (no Taylor) to reconstruct "helloworld".
//
// only
// thread (0,0,0) prints to avoid spam.
// ---------------------------------------------------------------------
__global__
void debugPrintKernel(long long CYC, const double* __restrict__ E)
{
    // Only one thread prints
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 ||
        threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0)
        return;

    int msg_len = zs;

    // We only start printing once enough letters have been injected
    if (CYC < msg_len - 1) return;

    int Z = blockIdx.z;
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z_cur = (int)(CYC % zs);

    // Reconstruct time at 2.0 using VAR_T derivatives at z_cur
    printf("CYC=%2lld  t_cur=%4.1f  t_back=%4.1f  msg_back=",CYC, ((double)CYC), reconstructTemporalValueForOrder(E,/*k0=*/0,VAR_T, Z, x, y, z_cur,(( 2.0 )-((double)CYC)),/*max_order_use=*/Xs - 1));

    // Reconstruct "hello" from z-history of VAR_MSG:
    // tick j was stored at z_j = j % zs (no wrap for CYC <= 9, zs=10)
    for (int j = 0; j < msg_len; ++j) {
        int z_j = j % zs;
        char c = (char)(int)lrint( E[UID_6D(0, VAR_MSG, Z, x, y, z_j)] );   // convert stored double back to char
        if (c == 0) c = '_';            // for safety / visualization
        printf("%c", c);
    }
    printf("\n");
}

// ---------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------
int main()
{
    double *d_E = nullptr;
    CUDA_CHECK(cudaMalloc(&d_E, TOTAL_E * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_E, 0, TOTAL_E * sizeof(double)));

    // We only use one spatial point and one Y-plane per kernel, but the layout is 6D.
    dim3 blocks(1, Ys, Zs);  // only Z=0
    dim3 threads(xs, ys, zs);

    double dt = 1.0;

    for (long long CYC = 0; CYC <= 25; ++CYC) {
        injectMsgHistoryKernel<<<blocks, threads>>>(CYC, d_E);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        fillKsKernel<<<blocks, threads>>>(CYC,dt,d_E);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        debugPrintKernel<<<blocks, threads>>>(CYC, d_E);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaFree(d_E));
    return 0;
}
