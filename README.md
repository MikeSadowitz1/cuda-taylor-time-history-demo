# cuda-taylor-time-history-demo
A 6D Taylor-series + ring-buffer demo in CUDA (“hello world” through derivatives)

# Taylor 6D CUDA "Hello World"

This project is a small but fully 6-dimensional CUDA demo that combines:

- A **6D flattened tensor layout**: `(X, Y, Z, x, y, z)` → single linear index  
- **Taylor-series reconstruction** of a continuous analytic variable (time)  
- A **discrete message channel** (`VAR_MSG`) stored in a ring buffer along the `z` (history) dimension  
- A debug kernel that reconstructs both **time** and a **"hello world"** message from the stored history

It’s essentially a toy “physics/fields + message” engine that shows how to mix **continuous derivatives** (good for Taylor) with **discrete events** (good for direct history lookup) in one unified 6D state array on the GPU.

---

## Core Ideas

### 1. 6D Indexing: `UID_6D`

All state lives in a single flat array `E` of length:

TOTAL_E = Xs * Ys * Zs * xs * ys * zs;
The helper:
long long UID_6D(int X, int Y, int Z, int x, int y, int z);
maps the logical 6D coordinates:

X – derivative order

Y – variable index (VAR_T for time, VAR_MSG for message)

Z – coarse spatial dimension

x, y – fine spatial coordinates

z – time-history index (ring buffer depth)

into a single linear offset in E. This lets you treat each point in space + history as a bundle of derivatives for each variable.

2. Analytic Time via Taylor-Series
For the continuous time variable VAR_T, we store the derivatives such that:

E^0_t(t) = t

E^1_t(t) = 1

E^k_t(t) = 0 for k >= 2

Given these derivatives at a current time t, the function
double reconstructTemporalValueForOrder(
    const double* E,
    int k0,
    int Y, int Z, int x, int y, int z,
    double h,
    int max_order_use
);
uses a Taylor series:

E(k0​)(t+h)≈m=0∑M​m!hm​E(k0​+m)(t)
to reconstruct the value at a target time t + h. Because the derivatives for VAR_T are exact, the reconstruction of time is exact as well.

3. Discrete Message Channel with Ring Buffer
For the discrete message variable VAR_MSG, we don’t pretend it’s smooth. Instead, we:

Inject characters over cycles CYC = 0..9:
0 → 'h'
1 → 'e'
2 → 'l'
3 → 'l'
4 → 'o'
5 → 'w'
6 → 'o'
7 → 'r'
8 → 'l'
9 → 'd'
Store the active character in the current z slice, where
int z_cur = CYC % zs;
This means the z dimension acts as a ring buffer / sliding window in time:

New data always writes into z_cur

Older values reside in the other z slots

The history naturally wraps once CYC >= zs

We never apply Taylor to VAR_MSG. Instead, we reconstruct the message by directly reading past history slices.

4. Debug Kernel: Printing Time + "hello world"
debugPrintKernel runs as a single-thread inspection kernel:

Reconstructs a target time using Taylor on VAR_T

Reconstructs the "helloworld" string by scanning the VAR_MSG history along z

Once enough cycles have passed to fill the message, it prints lines like:

CYC= 9  t_cur= 9.0  t_back= 2.0  msg_back=helloworld
The msg_back string is built from the characters stored in the ring buffer.

Code Structure
UID_6D(...)
6D → 1D index mapping for the unified state array E.

reconstructTemporalValueForOrder(...)
Device function implementing Taylor-series reconstruction for continuous variables (used for VAR_T).

injectMsgHistoryKernel(...)
Injects discrete message characters into the VAR_MSG variable at the current history index z_cur.

fillKsKernel(...)
Fills derivatives for:

VAR_T — analytic time variable with exact derivatives

VAR_MSG — estimates first and second time derivatives from discrete history (finite differences).

debugPrintKernel(...)
Single-thread kernel that prints:

Current cycle CYC

Reconstructed time at a target point

Reconstructed "helloworld" from the VAR_MSG history.

main()
Allocates and zeroes the GPU buffer, sets up grid/block dimensions, runs the update loop for CYC = 0..25, and launches the three kernels in sequence each step.

Building and Running
Requirements
CUDA-capable GPU

CUDA Toolkit (e.g. 11.x or later)

A C++ compiler supported by your CUDA version

Compile
bash:
nvcc -O2 -o ctthd ctthd.cu

Run
bash:
./ctthd
You should see lines printed once enough history has accumulated, including a reconstructed "helloworld" from the VAR_MSG history.

Why This Might Be Interesting
Demonstrates multi-dimensional indexing and layout for a “field of derivatives” in CUDA.

Shows how to mix analytic Taylor reconstruction with discrete event history, using a ring buffer along time.

Serves as a conceptual stepping stone toward more complex PDE solvers, neural fields, or time-history-based models that live on the GPU.

Feel free to fork, modify, or adapt this as a sandbox for your own time-history + Taylor experiments.
