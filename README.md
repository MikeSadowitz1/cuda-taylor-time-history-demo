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


# GPU Persistent Kernel Demo  
### 6D Field Simulation + Live Message Streaming

This project demonstrates a **multi-block persistent CUDA kernel** capable of processing a 6-dimensional data field while the host streams new updates each cycle.  
Communication uses **mapped pinned memory** so the GPU never has to relaunch.

The demo includes:

- A fully persistent kernel (runs until `quit=1`)
- A custom **device-wide barrier** built from atomics
- Temporal derivative reconstruction for the `VAR_T` field
- A streaming message system storing characters in the **(x,z) plane**
- Host-side and device-side reconstructions of stored data
- Pretty printers to visualize message layout

---

## Data Model: 6D Field

The global array:

E[Xs][Ys][Zs][xs][ys][zs]

is flattened using `UID_6D()`.  

Dimensions:

Xs = 10 (temporal derivative order)
Ys = 10 (variable class: VAR_T or VAR_MSG)
Zs = 10
xs = 10
ys = 10
zs = 10 (time-ring buffer)

`UID_6D()` maps the full 6D coordinate into a linear offset in `E`.

---

## Message Packing (x,z Plane)

Messages are stored using:

global_idx = x * zs + z

Meaning:

- Each column is a time slice (`z`)
- Each row (`x`) increments the major index
- Maximum message length = `xs * zs` (100 chars here)

If the message is too long, the host prints:

ERROR: not enough xs*zs to store full string

and the extra characters are ignored.

---

## Persistent Kernel Logic

The kernel:

1. Starts once and **never exits** until `d_quit = 1`
2. Polls `d_CYC` for a new cycle index
3. Uses a device-wide barrier composed of:
   - `g_activeCycle`
   - `g_threadsDone`
   - `atomicAdd` + volatile + fencing
4. Each thread processes a strided subset of tasks:
for t = tid; t < TOTAL_TASKS; t += numThreads

5. After all work is done:
- last thread prints GPU debug reconstruction
- last thread sets `*d_done = CYC` with `__threadfence_system()`

This enables a full persistent-kernel pipeline without cooperative launch.

---

## Temporal Reconstruction

`reconstructTemporalValueForOrder()` evaluates a Taylor-like expansion using the stored derivatives in E:

E[0] = value
E[1] = first derivative
E[2] = second derivative
...

Used for debugging and demonstration.

---

## Host-Side Debugging Tools

Two helpers allow easy visualization:

### 1. `dumpMsgLayout(h_E)`
Prints an ASCII matrix showing the entire `(x,z)` layout.

### 2. `reconstructFullMessage(h_E)`
Rebuilds the full message (row-major in `x`, then `z`).

Example output:

=== Host-side MSG layout ===
x=0: hello_____
x=1: world_____
...
Full reconstructed message (host): [helloworldwhatsgoing]

---

## Build Instructions

Requires:

- CUDA-capable GPU supporting mapped pinned memory
- Compute capability 7.0+ recommended (sm_70 / sm_75 / sm_80)

Compile with:

nvcc -O3 -std=c++17 -arch=sm_80 ctthd.cu -o ctthd


(Adjust `sm_XX` to match your GPU.)

---

## Runtime Flow

1. Host initializes pinned-mapped arrays
2. Host launches persistent kernel
3. Loop over `CYC = 0 .. CYCS_TO_RUN`:
   - Host writes new message characters into `h_E`
   - Host sets `h_CYC = CYC`
   - Device completes the cycle, sets `d_done`
   - Host waits for `h_done == CYC`
4. Host sets `quit = 1`, GPU exits
5. Host prints final layout & reconstructed string

---

## Example Kernel Output

GPU debug: CYC=09 t_cur=9.0 t_back=...
helloworldwhat

Then at program end:

=== Host-side MSG layout (Z=0,y=0) ===
x=0: hello_____
x=1: world_____
x=2: whats_____
...
Full reconstructed message (host): [helloworldwhatsgoing]


---

## License

This project is provided for educational and experimental purposes only.  
No warranty expressed or implied.




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

