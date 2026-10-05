import numpy as np
from scipy.optimize import brentq

# ---- Flow / geometry inputs ----
r_inlet  = 3.0    # in, converging inlet radius
r_throat = 1.25   # in, throat radius
r_exit   = 2.5    # in, exit radius
n_conv   = 20     # points in converging section
n_div    = 30     # points in diverging section
gamma    = 1.2    # ratio of specific heats
p0       = 300.0  # psi, chamber (stagnation) pressure

# ---- Carbon fiber: constituents (rule of mixtures) ----
E_fiber  = 33.4e6   # psi, fiber modulus (T300-class standard modulus)
E_matrix = 0.5e6    # psi, epoxy modulus
V_f      = 0.45     # fiber volume fraction (typical hand wet layup: 0.35-0.50)
k_weave  = 0.5      # fraction of fibers running in the hoop direction (balanced 0/90 weave)

# ---- Carbon fiber: design limits ----
sigma_ult = 60000.0   # psi, hoop tensile strength (CHANGE to coupon data)
SF        = 3.0       # safety factor on strength
eps_allow = 0.003     # allowable hoop strain (0.3%) to limit matrix microcracking / leaks
t_ply     = 0.010     # in, cured thickness per ply

sigma_allow = sigma_ult / SF
A_star = np.pi * r_throat**2

# ---- Laminate hoop modulus (rule of mixtures) ----
def hoop_modulus(Ef, Em, Vf, k):
    E_ud = Vf * Ef + (1 - Vf) * Em   # modulus along the fibers (unidirectional)
    return k * E_ud                  # only fraction k of fibers point in the hoop direction

E_hoop = hoop_modulus(E_fiber, E_matrix, V_f, k_weave)

# ---- Area-Mach relation ----
def area_ratio(M, g=gamma):
    return (1/M) * ((2/(g+1)) * (1 + (g-1)/2 * M**2))**((g+1)/(2*(g-1)))

def mach_subsonic(ar):
    if np.isclose(ar, 1.0):
        return 1.0
    return brentq(lambda M: area_ratio(M) - ar, 1e-6, 0.9999)

def mach_supersonic(ar):
    if np.isclose(ar, 1.0):
        return 1.0
    return brentq(lambda M: area_ratio(M) - ar, 1.0001, 20)

# ---- Isentropic pressure relation ----
def pressure_ratio(M, g=gamma):
    return (1 + (g-1)/2 * M**2) ** (-g/(g-1))

# ---- Wall sizing ----
def thickness_strength(p, r):
    return p * r / sigma_allow            # t so that stress <= sigma_allow

def thickness_stiffness(p, r):
    return p * r / (E_hoop * eps_allow)   # t so that strain <= eps_allow

def hoop_stress(p, r, t):
    return p * r / t

def hoop_strain(p, r, t):
    return p * r / (E_hoop * t)           # eps = sigma / E

def radial_growth(p, r, t):
    return hoop_strain(p, r, t) * r       # delta_r = eps * r

# ---- Run one section, one station at a time ----
def run_section(r_values, mach_solver):
    rows = []
    for r in r_values:
        A      = np.pi * r**2
        AR     = A / A_star
        M      = mach_solver(AR)
        p      = p0 * pressure_ratio(M)
        t_str  = thickness_strength(p, r)
        t_stf  = thickness_stiffness(p, r)
        t_req  = max(t_str, t_stf)                     # the stricter one wins
        governs = "strength" if t_str >= t_stf else "strain"
        plies  = int(np.ceil(t_req / t_ply))
        t_act  = plies * t_ply
        sigma  = hoop_stress(p, r, t_act)
        eps    = hoop_strain(p, r, t_act)
        dr     = radial_growth(p, r, t_act)
        rows.append([r, M, p, t_str, t_stf, governs, plies, sigma, eps, dr])
    return rows

def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'r [in]':>7} {'M':>6} {'p [psi]':>8} {'t_str':>7} {'t_stf':>7} {'governs':>9} "
          f"{'plies':>5} {'sigma':>7} {'strain%':>8} {'dr [in]':>8}")
    for r, M, p, t_str, t_stf, governs, plies, sigma, eps, dr in rows:
        print(f"{r:7.3f} {M:6.3f} {p:8.2f} {t_str:7.4f} {t_stf:7.4f} {governs:>9} "
              f"{plies:5d} {sigma:7.0f} {eps*100:8.3f} {dr:8.5f}")

print(f"E_hoop = {E_hoop/1e6:.2f} Msi   sigma_allow = {sigma_allow:.0f} psi   "
      f"strain-equivalent stress = {E_hoop*eps_allow:.0f} psi")

conv_rows = run_section(np.linspace(r_inlet, r_throat, n_conv), mach_subsonic)
print_table("CONVERGING SECTION (subsonic)", conv_rows)

div_rows = run_section(np.linspace(r_throat, r_exit, n_div), mach_supersonic)
print_table("DIVERGING SECTION (supersonic)", div_rows)
