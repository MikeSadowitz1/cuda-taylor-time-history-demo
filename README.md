# cuda-taylor-time-history-demo
A 6D Taylor-series + ring-buffer demo in CUDA (“hello world” through derivatives)
Now with a persistent kernel and streaming host updates

# Taylor 6D CUDA "Hello World"

This project is a small but fully 6-dimensional CUDA demo that combines:

- A **6D flattened tensor layout**: `(X, Y, Z, x, y, z)` → single linear index  
- **Taylor-series reconstruction** of a continuous analytic variable (time)  
- A **discrete message channel** (`VAR_MSG`) stored in a ring buffer along the `z` (history) dimension  
- A persistent CUDA kernel that runs in a while (true) loop and reacts to live host updates, that reconstructs both **time** and a **"hello world"** message from the stored history
- A simple host↔device handshake that streams characters into the GPU over “cycles”

It’s essentially a toy “physics/fields + message” engine that shows how to mix **continuous derivatives** (good for Taylor) with **discrete events** (good for direct history lookup) & Uses a persistent GPU kernel as a long-lived worker that watches shared state and updates continuously as the host streams new data.
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

4. Persistent Kernel + Streaming From the Host

The original version launched three kernels per time step:

injectMsgHistoryKernel()

fillKsKernel()

debugPrintKernel()

The updated version introduces a persistent kernel:

__global__
void persistentKernel(double* E,
                      volatile int* d_cycle,
                      volatile int* d_quit,
                      volatile int* d_done)
{
    __shared__ int lastCycle;
    // initialize lastCycle...

    while (true) {
        // 1) Check for quit signal
        // 2) Read the current cycle from host-mapped memory
        // 3) If new cycle, update derivatives in E
        // 4) Optionally print debug output ("hello world")
        // 5) Signal back to the host that this cycle is done
    }
}

How it works

The host and device share a small set of control integers in mapped pinned memory:

cycle – which cycle the kernel should process

done – last cycle fully processed by the kernel

quit – flag to tell the kernel to exit

The host loop does:

Write the next character(s) into the host-side h_E buffer

Update h_cycle = cycle

Spin until h_done >= cycle (i.e., the GPU has finished that cycle)

Move on to the next cycle

The persistent kernel:

Reads *d_cycle (via volatile pointer)

If curCycle != lastCycle, it:

Fills analytic time derivatives for VAR_T

Computes finite-difference derivatives for VAR_MSG

Optionally prints a debug line once enough history is present

Updates lastCycle and writes *d_done = curCycle so the host knows it’s done

If *d_quit != 0, it breaks out of the loop and returns

This pattern turns the kernel into a long-lived GPU worker that reacts to streaming host updates instead of being relaunched every time step.

5. Debug Printing: Time + "helloworld"

Inside the persistent kernel, a single designated thread (block (0,0,0), thread (0,0,0)) acts as a “debug inspector”:

Reconstructs time at a target point using Taylor on VAR_T

Reconstructs the "helloworld" message from the VAR_MSG ring buffer

Once enough cycles have accumulated to fill the string, it prints lines like:

CYC= 9  t_cur= 9.0  t_back= 2.0  msg_back=helloworld
CYC=10  t_cur=10.0  t_back= 2.0  msg_back=helloworld
...


msg_back is built by reading VAR_MSG at each history slot z_j = j % zs and converting the stored double back to char.

Code Structure
UID_6D(...)
6D → 1D index mapping for the unified state array E.
reconstructTemporalValueForOrder(...)
Device function implementing Taylor-series reconstruction for continuous variables (used for VAR_T).
persistentKernel(...) (new)


Runs as a persistent, while(true) kernel


Watches shared state (cycle, quit, done) via volatile pointers


For each new cycle:


Updates analytic time derivatives (VAR_T)


Updates discrete message derivatives (VAR_MSG) via finite differences


Prints debug line once the "helloworld" sequence is visible


Signals completion to the host




Host-Side Streaming Loop


Uses pinned, mapped host memory for:


E (the big 6D state array)


cycle, done, quit (control variables)




Each iteration:


Injects one character of "helloworld" into the ring buffer


Publishes a new cycle


Waits until the persistent kernel acknowledges it processed that cycle





Building and Running
Requirements


CUDA-capable GPU


CUDA Toolkit (e.g. 11.x or later)


A C++ compiler supported by your CUDA version


Compile
nvcc -O2 -o ctthd ctthd.cu

Run
./ctthd

You should see lines printed once enough history has accumulated, including a reconstructed "helloworld" from the VAR_MSG history, driven by the persistent streaming kernel rather than per-step launches.

Why This Might Be Interesting


Demonstrates multi-dimensional indexing and layout for a “field of derivatives” in CUDA.


Shows how to combine:


Analytic Taylor reconstruction for continuous variables (VAR_T), and


Direct history lookups / finite differences for discrete variables (VAR_MSG),


within a single 6D state array.


Serves as a conceptual stepping stone toward:


Persistent-kernel designs for real-time or streaming systems


PDE solvers or neural fields with time history on the GPU


Low-latency ML / physics engines where the host continuously streams updates into a long-running GPU worker




Feel free to fork, modify, or adapt this as a sandbox for your own time-history + Taylor + persistent kernel experiments.

