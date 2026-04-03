#!/usr/bin/env python3
"""
OMNI-ENGINE v51.0: TCK (CHRONON STATE AUTOMATA)
===============================================
Absolutní zjednodušení reality na ternární stavový automat (0-1-2-3).

LOGIKA HARDWARU:
- Tik mřížky = 1 Chronon (Planckův čas).
- Každý uzel Ω dýchá fázové bity.
- Hmota (Stav 2) vzniká tam, kde se puls "zasukuje".
- Sifon (U) je hardwarový limit mřížky, který ořezává Stav 3 zpět na 2.

Architekt: Rudolf Bandor & TIK Heuristic Engine
"""

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
from scipy.ndimage import gaussian_filter, zoom

# ============================================================================
# TCK MASTER CONSTANTS (GEOMETRY LOCK)
# ============================================================================
PHI = (1.0 + np.sqrt(5.0)) / 2.0
PHI_SQ = PHI**2                          # Sigma = 2.618 (Mez pevnosti)
ALPHA_INV = 137.036                      # Bandwidth Ω
KAPPA_P = 178.435                        # Protonová rezonance
PSI_H = 138.20                           # Rezonanční kotva Vodíku

# --- KONFIGURACE ---
N = 128
VIEW_SIZE = 40                           # Fokus na živý uzel
UP_SCALE = 15                            # Vyhlazení pro Level A zobrazení

# ============================================================================
# OPENCL KERNEL: CHRONON STATE ENGINE
# ============================================================================
kernel_code = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__constant double PHI_SQ = 2.61803398875;
__constant double KAPPA_P = 178.435;
__constant double LAMBDA_V = 0.0309;
__constant double XI_AMP = 0.0191;

// Ternární chrononový generátor {-1, 0, 1}
double get_chronon_bit(int x, int y, int z, int t, int s) {
    long n = x*73856093 ^ y*19349663 ^ z*83492791 ^ t*67228037 ^ s;
    n = (n >> 13) ^ n;
    int bit = (int)(abs(n) % 3) - 1;
    return (double)bit * XI_AMP;
}

__kernel void chronon_automata_step(
    __global double *psi_r, __global double *psi_i, __global double *h_hm,
    const int grid_N, const double t_glob,
    const double q1x, const double q1y, const double q1z,
    const double q2x, const double q2y, const double q2z,
    const double q3x, const double q3y, const double q3z,
    const double dt, const int seed)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    if (x >= grid_N || y >= grid_N || z >= grid_N) return;
    int i = x * grid_N * grid_N + y * grid_N + z;

    double pr = psi_r[i]; double pi = psi_i[i];
    
    // 1. 6D LAPLACIÁN Ω (Hardware mřížky)
    double lap_r = -12.0 * pr;
    int neighbors[12][3] = {{0,1,2},{1,2,0},{2,0,1},{0,1,-2},{1,-2,0},{-2,0,1},
                            {0,-1,2},{-1,2,0},{2,0,-1},{0,-1,-2},{-1,-2,0},{-2,0,-1}};
    for(int k=0; k<12; k++) {
        int nx = (x + neighbors[k][0] + grid_N) % grid_N;
        int ny = (y + neighbors[k][1] + grid_N) % grid_N;
        int nz = (z + neighbors[k][2] + grid_N) % grid_N;
        lap_r += psi_r[nx * grid_N * grid_N + ny * grid_N + nz];
    }

    // 2. SUKOVÁNÍ DO 3 (Nukleární tendence)
    double d1 = max(sqrt(pow((double)x-q1x,2)+pow((double)y-q1y,2)+pow((double)z-q1z,2)), 0.3);
    double d2 = max(sqrt(pow((double)x-q2x,2)+pow((double)y-q2y,2)+pow((double)z-q2z,2)), 0.3);
    double d3 = max(sqrt(pow((double)x-q3x,2)+pow((double)y-q3y,2)+pow((double)z-q3z,2)), 0.3);
    double psi_knot = (KAPPA_P / 3.0) * (1.0/pow(d1, 4.0) + 1.0/pow(d2, 4.0) + 1.0/pow(d3, 4.0));

    // 3. EVOLUCE STAVU (Strojový kód mřížky)
    double chronon = get_chronon_bit(x, y, z, (int)t_glob, seed);
    
    // nr = současný stav + difuze mřížky + sukování jádra + dopadající chronon
    double nr = (pr + 0.0833 * lap_r * dt + psi_knot * 0.15 * dt + chronon) * (1.0 - LAMBDA_V * dt);
    double ni = (pi + chronon) * (1.0 - LAMBDA_V * dt);

    // 4. SIFON (Mechanika ořezu 3 -> 2)
    double M = sqrt(nr*nr + ni*ni);
    if (M >= PHI_SQ) {
        // Mřížka 'odpustí' přetlak do 8D stínu
        double reduction = PHI_SQ / M;
        nr *= reduction;
        ni *= reduction;
        M = PHI_SQ;
    }

    psi_r[i] = nr; psi_i[i] = ni;
    h_hm[i] = M;
}
"""

class ChrononAutomataEngine:
    def __init__(self, size=128):
        self.size = size
        self.dt = 0.2 # Takt procesoru
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernel_code).build()
        self.knl = cl.Kernel(self.prg, "chronon_automata_step")
        
        self.d_pr = cl_array.zeros(self.queue, size**3, np.float64)
        self.d_pi = cl_array.zeros(self.queue, size**3, np.float64)
        self.d_hm = cl_array.zeros(self.queue, size**3, np.float64)
        
        # Osvobozené solitony (Suky jádra)
        center = size / 2
        self.q_pos = np.array([[center+2, center, center], [center-1, center+1.7, center], [center-1, center-1.7, center]])
        self.q_vel = np.zeros_like(self.q_pos)
        
        self.setup_viz()

    def setup_viz(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10), facecolor='#020202')
        self.im = self.ax.imshow(np.zeros((VIEW_SIZE * UP_SCALE, VIEW_SIZE * UP_SCALE)), 
                                  cmap='magma', origin='lower',
                                  norm=colors.PowerNorm(0.3, 0.1, 280.0))
        self.ax.axis('off')
        self.txt = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, color='cyan', family='monospace')
        self.ax.set_title("TCK v51.0: CHRONON STATE AUTOMATA", color='white', pad=20)

    def update_solitons(self):
        core_c = np.mean(self.q_pos, axis=0)
        for i in range(3):
            r = self.q_pos[i] - core_c
            d = np.linalg.norm(r)
            # 1. Silná vazba (Vnitřní sukování)
            self.q_vel[i] -= r * 0.08
            # 2. Topologický vír (Fázový moment)
            vort = np.array([-r[1], r[0], 0])
            self.q_vel[i] += (vort / (d + 0.1)) * 0.18
            # 3. Chrononové nárazy (Ternární motor)
            self.q_vel[i] += np.random.choice([-1, 0, 1], (3,)) * 0.14
            self.q_vel[i] *= 0.94
            self.q_pos[i] += self.q_vel[i] * self.dt
        # Jemná stabilizace k nulovému bodu mřížky
        self.q_pos += (self.size/2 - np.mean(self.q_pos, axis=0)) * 0.01

    def run(self):
        print(f"[*] TCK AUTOMATA v51.0 | 1 TIK = 1 CHRONON | SIPHON ACTIVE")
        t = 0
        v_h = VIEW_SIZE // 2
        center_f = self.size // 2
        
        while True:
            self.update_solitons()
            qs = self.q_pos.flatten()
            
            self.knl(self.queue, (self.size, self.size, self.size), None,
                     self.d_pr.data, self.d_pi.data, self.d_hm.data,
                     np.int32(self.size), np.float64(t), *np.float64(qs),
                     np.float64(self.dt), np.int32(time.time()+t))

            if t % 6 == 0:
                hm_vol = self.d_hm.get().reshape((self.size, self.size, self.size))
                roi = hm_vol[center_f-v_h : center_f+v_h, center_f-v_h : center_f+v_h, :]
                mip_2d = np.max(roi, axis=2).T
                
                hd_img = zoom(mip_2d, UP_SCALE, order=1)
                self.im.set_data(gaussian_filter(hd_img, sigma=0.6))
                
                self.txt.set_text(
                    f"TCK v51.0 | CHRONON MACHINE\n"
                    f"---------------------------\n"
                    f"STATE 0-1-2: ACTIVE\n"
                    f"SIPHON (3->2): {('ENGAGED' if np.max(mip_2d) >= PHI_SQ else 'IDLE')}\n"
                    f"CORE REZ: {np.max(mip_2d):.1f}\n"
                    f"TIK: {t}"
                )
                self.fig.canvas.draw(); plt.pause(0.01)
            t += 1

if __name__ == "__main__":
    engine = ChrononAutomataEngine(size=128)
    engine.run()
