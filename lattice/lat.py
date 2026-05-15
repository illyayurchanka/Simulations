"""
Von Kármán Vortex Street – D2Q9 Lattice Boltzmann Method
Numba-accelerated implementation.
"""

from numba import njit, prange
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os, time

# ─── Parameters (from slides) ────────────────────────────────────────────────
nx, ny    = 520, 180        # grid size
uLB       = 0.04            # inlet lattice velocity
Re        = 220.0           # Reynolds number
cx, cy    = nx // 4, ny // 2   # cylinder centre  (≈ 130, 90)
r         = 9               # cylinder radius
nulb      = uLB * r / Re   # lattice kinematic viscosity
tau       = 3.0 * nulb + 0.5   # BGK relaxation time

print(f"Grid      : {nx} × {ny}")
print(f"Cylinder  : centre=({cx},{cy}), radius={r}")
print(f"Re={Re:.0f},  u_LB={uLB},  ν_LB={nulb:.5f},  τ={tau:.5f}")

# ─── D2Q9 lattice ─────────────────────────────────────────────────────────────
c      = np.array([(x,y) for x in [0,-1,1] for y in [0,-1,1]], dtype=np.int32)
# → [ [0,0],[0,-1],[0,1],[-1,0],[-1,-1],[-1,1],[1,0],[1,-1],[1,1] ]
tw     = np.array([4/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/9, 1/36, 1/36])
noslip = np.array([c.tolist().index((-c[i]).tolist()) for i in range(9)],
                  dtype=np.int32)
print(f"noslip    : {noslip.tolist()}")   # [0,2,1,6,8,7,3,5,4]

# ─── Obstacle mask ────────────────────────────────────────────────────────────
obstacle = np.fromfunction(
    lambda x, y: (x-cx)**2 + (y-cy)**2 < r**2, (nx, ny)).astype(np.bool_)

# ─── Numba-JIT core step ──────────────────────────────────────────────────────
@njit(parallel=True)
def lbm_step(fin, fout, c, tw, tau, noslip, obstacle, vel0):
    """
    One LBM time step (in-place).
    fin/fout : (9, nx, ny)  population arrays
    vel0     : (ny,)        prescribed x-velocity at inlet
    Returns  : u (2,nx,ny), rho (nx,ny)
    """
    q, nx_, ny_ = fin.shape

    # ── 1. Outlet BC (right wall x=nx-1) – zero-gradient outflow ────────────
    for b in range(ny_):
        fin[3, -1, b] = fin[3, -2, b]
        fin[4, -1, b] = fin[4, -2, b]
        fin[5, -1, b] = fin[5, -2, b]

    # ── 2. Macroscopic density & velocity ────────────────────────────────────
    rho = np.zeros((nx_, ny_))
    u   = np.zeros((2, nx_, ny_))
    for x in prange(nx_):
        for y in range(ny_):
            s = 0.0
            for i in range(q):
                s += fin[i, x, y]
            rho[x, y] = s
            for d in range(2):
                su = 0.0
                for i in range(q):
                    su += c[i, d] * fin[i, x, y]
                u[d, x, y] = su / s

    # ── 3. Inlet BC (left wall x=0) – Zou–He velocity BC ────────────────────
    for b in range(ny_):
        u[0, 0, b] = vel0[b]
        u[1, 0, b] = 0.0
        known = (fin[0,0,b] + fin[1,0,b] + fin[2,0,b]
                 + 2.0*(fin[3,0,b] + fin[4,0,b] + fin[5,0,b]))
        rho[0, b] = known / (1.0 - vel0[b])

    # ── 4. Equilibrium + BGK collision + bounce-back ─────────────────────────
    for x in prange(nx_):
        for y in range(ny_):
            usq = u[0,x,y]**2 + u[1,x,y]**2
            for i in range(q):
                cu = c[i,0]*u[0,x,y] + c[i,1]*u[1,x,y]
                feq_i = rho[x,y] * tw[i] * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usq)
                if obstacle[x, y]:
                    # Bounce-back: reflect incoming population
                    fout[i, x, y] = fin[noslip[i], x, y]
                elif x == 0 and (i == 6 or i == 7 or i == 8):
                    # Inlet unknown populations set to equilibrium
                    fout[i, x, y] = feq_i
                else:
                    fout[i, x, y] = fin[i,x,y] - (fin[i,x,y] - feq_i) / tau

    # ── 5. Streaming (periodic; inlet/outlet override next step) ─────────────
    for i in range(q):
        dx = c[i, 0]
        dy = c[i, 1]
        for x in prange(nx_):
            for y in range(ny_):
                xn = (x + dx) % nx_
                yn = (y + dy) % ny_
                fin[i, xn, yn] = fout[i, x, y]

    return u, rho


# ─── Numpy equilibrium (for initialisation only) ──────────────────────────────
def equilibrium_np(rho, u):
    cu  = np.einsum('ia,axy->ixy', c.astype(float), u)
    usq = u[0]**2 + u[1]**2
    return (rho[np.newaxis] * tw[:,np.newaxis,np.newaxis]
            * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*usq[np.newaxis]))

# ─── Initialisation ───────────────────────────────────────────────────────────
y_idx = np.arange(ny, dtype=np.float64)
vel0  = uLB * (1.0 + 1e-4 * np.sin(y_idx / ny * 2.0 * np.pi))  # (ny,)

vel_2d         = np.zeros((2, nx, ny))
vel_2d[0, :, :] = uLB
feq = equilibrium_np(np.ones((nx, ny)), vel_2d)
fin = feq.copy()
fout = fin.copy()

# ─── JIT warm-up ──────────────────────────────────────────────────────────────
print("Compiling Numba JIT kernel …", end=' ', flush=True)
t_compile = time.time()
lbm_step(fin, fout, c, tw, tau, noslip, obstacle, vel0)
print(f"done ({time.time()-t_compile:.1f}s)")

# ─── Main loop ────────────────────────────────────────────────────────────────
max_steps    = 20000
frames_saved = 0

t0 = time.time()
for step in range(max_steps + 1):
    u, rho = lbm_step(fin, fout, c, tw, tau, noslip, obstacle, vel0)

    if step % 100 == 0:
        plt.clf()
        plt.imshow(np.sqrt(u[0]**2 + u[1]**2), cmap=cm.Reds)
        plt.savefig(f"vel.{str(step // 100).zfill(4)}.png")
        frames_saved += 1

        elapsed = time.time() - t0
        rate    = (step + 1) / elapsed if elapsed > 0 else 0
        eta     = (max_steps - step) / rate if rate > 0 else 0
        print(f"  step {step:6d}/{max_steps}  |  {elapsed:6.1f}s  |  "
              f"{rate:.1f} steps/s  |  ETA {eta:.0f}s")

print(f"\nFinished — {frames_saved} frames in ./frames/")
print(f"Total wall time: {time.time()-t0:.1f}s")

# ─── GIF creation ─────────────────────────────────────────────────────────────
print("\nBuilding animated GIF …")
from PIL import Image
import glob

gif_path   = './frames/outputs/vortex_street.gif'
gif_width  = 900          # resize to this width to keep file size reasonable
ms_per_frame = 80         # 80 ms ↔ ~12 fps

frame_paths = sorted(glob.glob('vel.*.png'))

gif_frames = []
for path in frame_paths:
    img = Image.open(path).convert('RGB')
    w, h = img.size
    new_h = int(h * gif_width / w)
    img   = img.resize((gif_width, new_h), Image.LANCZOS)
    gif_frames.append(img)

gif_frames[0].save(
    gif_path,
    save_all      = True,
    append_images = gif_frames[1:],
    duration      = ms_per_frame,   # milliseconds per frame
    loop          = 0,              # 0 = loop forever
    optimize      = False,
)

size_mb = os.path.getsize(gif_path) / 1e6
print(f"GIF saved → {gif_path}")
print(f"  {len(gif_frames)} frames  |  {ms_per_frame} ms/frame  |  {size_mb:.1f} MB")
