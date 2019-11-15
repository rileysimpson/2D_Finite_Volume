# Solves the 2D Laplace equation d^2(phi)/dx^2 + d^2(phi)/dy^2 = 0 with dirichlet boundary conditions
# finite volume formulation
# 2nd order accurate centered interpolation of interior nodes
# 1st order accurate interpolation of bounding nodes
# solved via Gauss-Seidel iteration

# Input:
# -Fluent Mesh File (ascii)
# -User settings (see below)

# Output:
# -2D mesh plot
# -Convergence plot
# -Solution contour plot
# -Solution 3D plots (cell centroids and vertices)
# -Residuals 3D plot (cell centroids)

# import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from mpl_toolkits.mplot3d import axes3d

# print setup
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

# ===input===
# mesh file
num_header = 12  # number of header lines
mesh_filename = "mesh2.msh"

# boundary conditions
north_BC = 0  # north fixed-value boundary condition
south_BC = 1  # south fixed-value boundary condition
east_BC = 0  # east fixed-value boundary condition
west_BC = 0  # west fixed-value boundary condition

# solver properties
norm_res_target = 0.001  # normalized residual target
num_n2 = 5  # number of inner iterations per outer iteration

# display options
pre_info = False
post_info = False
plot_contour = True
plot_mesh = True
plot_mesh_vectors = False
plot_mesh_nodes = False
plot_residual = True
plot_3D_residual = True
plot_cell = False
plot_vertex = True
n1n2_per_print = 20  # number of (inner * outer) iterations per print to screen
scale_unit_vectors = 1 / 8  # mesh plot unit vector scaling
contour_res = 100  # number of contour bands
# ===end input===

# === mesh setup ===
# ---load mesh---
# read mesh file into line separated character array
with open(mesh_filename, "r") as f:
    mesh = f.readlines()

# collect metadata
num_vertex = int((mesh[8].rsplit(' '))[3], 16)
num_face = int((mesh[9].rsplit(' '))[3], 16)
num_cell = int((mesh[10].rsplit(' '))[3], 16)
num_ivertex = int((mesh[11].rsplit(' '))[3], 16)

# load internal vertices
v_x = np.zeros(num_vertex)
v_y = np.zeros(num_vertex)
for i in range(0, num_ivertex):
    v_x[i] = ((mesh[i + num_header]).split())[0]
    v_y[i] = ((mesh[i + num_header]).split())[1]

# load boundary vertices
num_bvertex = num_vertex - num_ivertex
line_offset = num_header + num_ivertex + 2
for i in range(0, num_bvertex):
    v_x[i + num_ivertex] = ((mesh[i + line_offset]).split())[0]
    v_y[i + num_ivertex] = ((mesh[i + line_offset]).split())[1]

# setup face connectivity
f_mesh = np.zeros((num_face, 4))  # [v-0, v-1, right-cell, left-cell]
line_offset = num_header + num_vertex + 4
num_iface = int((mesh[line_offset - 1].rsplit(' '))[3], 16)

# internal face connectivity
for j in range(0, num_iface):
    for i in range(0, 4):
        f_mesh[j, i] = int(((mesh[j + line_offset]).split())[i], 16)

# boundary face connectivity
num_bface = num_face - num_iface
line_offset = num_header + num_vertex + num_iface + 7
for j in range(0, num_bface):
    for i in range(0, 4):
        f_mesh[j + num_iface, i] = int(((mesh[j + line_offset]).split())[i], 16)
# ---end load mesh---

# ---declare arrays---
# declare coordinate arrays
c_x = np.zeros(num_cell)  # x coordinate of cell centroid = c_x(global cell index)
c_y = np.zeros(num_cell)  # y coordinate of cell centroid = c_y(global cell index)
f_x = np.zeros(num_face)  # x coordinate of face midpoint = f_x(global cell index)
f_y = np.zeros(num_face)  # y coordinate of face midpoint = f_y(global cell index)
# declare face property arrays
fn_x = np.zeros(num_face)  # x component of face normal = fn_x(global face index)
fn_y = np.zeros(num_face)  # y component of face normal = fn_y(global face index)
ft_x = np.zeros(num_face)  # x component of face tangent = ft_x(global face index)
ft_y = np.zeros(num_face)  # y component of face tangent = ft_y(global face index)
fn_sign = np.zeros((num_cell, 3))  # sign modifier (-1 for inward pointing normal)
area = np.zeros(num_face)  # area of face = area(global face index)
delta = np.zeros(num_face)  # face normal component of cell centroid joining vector = delta(global face index)
# declare volume property arrays
vol = np.zeros(num_cell)  # volume of cell = vol(global cell index)
# declare connectivity arrays
c_f = np.full((num_cell, 3), -1, dtype=int)  # global face index = c_f(cell index, local face index)
f_c = np.zeros((num_face, 2), dtype=int)  # global cell index = f_c(face index, local cell index)
f_v = np.zeros((num_face, 2), dtype=int)  # global vertex index = f_v(face index, local vertex index)
c_v = np.full((num_cell, 3), -1, dtype=int)  # global vertex index = c_v(cell index, local vertex index)
bf_f = np.zeros(num_bface, dtype=int)  # global face index = bf_f(boundary face index)
f_bf = np.zeros(num_face, dtype=int)  # boundary face index = bf_f(global face index)
# declare flag arrays
bf_flag = np.zeros(num_face, dtype=int)  # flagged (1) if boundary face, 0 for interior face
bv_flag = np.zeros(num_vertex, dtype=int)  # flagged (1) if boundary vertex, 0 for interior vertex
# declare link arrays
co_O = np.zeros(num_cell)  # coefficient of cell being iterated
co_nb = np.zeros((num_cell, 3))  # coefficient of neighbouring cells
# interpolation weights array
wt_c_f = np.zeros(num_face)  # cell-to-face interpolation weights, face_value=wt_c_f*cellA_value+(1-wt_c_f)*cellB_value
wt_c_v = np.zeros((num_cell, 3))  # cell-to-vertex interpolation weights, vertex_value=SUM(wt_c_v*cell_value)/wtsum_c_v
wtsum_c_v = np.zeros(num_vertex)  # Sum of cell-to-vertex interpolation weights for each vertex
# declare variable arrays
c_phi = np.zeros(num_cell)  # cell phi variable array
v_phi = np.zeros(num_vertex)  # vertex phi variable array
bphi = np.zeros(num_cell)  # boundary phi variable array
b_rhs = np.zeros(num_cell)  # boundary right-hand side contribution
wt_check = np.zeros(num_vertex)  # check cell-to-vertex interpolation weights, sum should be one
c_skew = np.zeros((num_cell, 3))  # tangential flux (skew) contribution of neighbour
c_skewsum = np.zeros(num_cell)  # sum of cell tangential flux (skew) contributions
c_r = np.zeros(num_cell)  # residual array of current timestep
c_r2 = np.zeros(2)  # RMS residual for each timestep
# ---end declare arrays---

# ---mesh connectivity & properties---
# face to vertex connectivity
for i in range(0, num_face):
    f_v[i, 0] = f_mesh[i, 0] - 1
    f_v[i, 1] = f_mesh[i, 1] - 1

# face to cell connectivity
for i in range(0, num_face):
    if f_mesh[i, 3] == 0:
        f_c[i, 0] = f_mesh[i, 2] - 1
        f_c[i, 1] = f_mesh[i, 2] - 1
    else:
        f_c[i, 0] = f_mesh[i, 2] - 1
        f_c[i, 1] = f_mesh[i, 3] - 1

# cell to face connectivity
# local face numbers are ordered by smallest to largest global number
for j in range(0, num_face):  # face index
    for i in range(0, 2):  # cell index
        c = f_c[j, i]  # get cell
        if c_f[c, 0] == -1:
            c_f[c, 0] = j
        elif c_f[c, 1] == -1 and c_f[c, 0] != j:  # one entry
            if j > c_f[c, 0]:  # global index larger than current
                c_f[c, 1] = j
            else:
                c_f[c, 1] = c_f[c, 0]  # global index less than current
                c_f[c, 0] = j
        elif c_f[c, 0] != j and c_f[c, 1] != j:  # two entries
            if j > c_f[c, 1]:  # global index larger than current
                c_f[c, 2] = j
            elif j > c_f[c, 0]:  # global index in-between than current
                c_f[c, 2] = c_f[c, 1]
                c_f[c, 1] = j
            else:  # global index less than current
                c_f[c, 2] = c_f[c, 1]
                c_f[c, 1] = c_f[c, 1]
                c_f[c, 0] = j

# cell to vertex connectivity
# local vertex numbers are ordered by smallest to largest global number
for k in range(0, num_face):  # face index
    for j in range(0, 2):  # cell index
        for i in range(0, 2):  # vertex index
            c = f_c[k, j]  # get cell
            v = f_v[k, i]  # get vertex
            if c_v[c, 0] == -1:  # no entries
                c_v[c, 0] = v
            elif c_v[c, 1] == -1 and c_v[c, 0] != v:  # one entry
                if v > c_v[c, 0]:  # global index larger than current
                    c_v[c, 1] = v
                else:
                    c_v[c, 1] = c_v[c, 0]  # global index less than current
                    c_v[c, 0] = v
            elif c_v[c, 0] != v and c_v[c, 1] != v:  # two entries
                if v > c_v[c, 1]:  # global index larger than current
                    c_v[c, 2] = v
                elif v > c_v[c, 0]:  # global index in-between than current
                    c_v[c, 2] = c_v[c, 1]
                    c_v[c, 1] = v
                else:  # global index less than current
                    c_v[c, 2] = c_v[c, 1]
                    c_v[c, 1] = c_v[c, 1]
                    c_v[c, 0] = v

# cell centroid coordinates
for i in range(0, num_cell):
    c_x[i] = (v_x[c_v[i, 0]] + v_x[c_v[i, 1]] + v_x[c_v[i, 2]]) / 3
    c_y[i] = (v_y[c_v[i, 0]] + v_y[c_v[i, 1]] + v_y[c_v[i, 2]]) / 3

# face midpoints
for i in range(0, num_face):
    v1 = f_v[i, 0]
    v2 = f_v[i, 1]
    f_x[i] = 1 / 2 * (v_x[v1] + v_x[v2])
    f_y[i] = 1 / 2 * (v_y[v1] + v_y[v2])

# boundary face connectivity and flag
for i in range(0, num_bface):
    bf_f[i] = i + num_iface
    bf_flag[i + num_iface] = 1
    f_bf[i + num_iface] = i

# surface normal
for i in range(0, num_face):
    v1 = f_v[i, 0]
    v2 = f_v[i, 1]
    dx = v_x[v2] - v_x[v1]
    dy = v_y[v2] - v_y[v1]
    area[i] = np.sqrt(dx ** 2 + dy ** 2)
    fn_x[i] = dy / area[i]
    fn_y[i] = -dx / area[i]
    ft_x[i] = dx / area[i]
    ft_y[i] = dy / area[i]

# average area
areasum = 0
for i in range(0, num_face):
    areasum = areasum + area[i]
area_avg = areasum / num_face

# domain dimensions
domain_x = 0
domain_y = 0
for i in range(0, num_bface):
    f = bf_f[i]
    if f_x[f] > domain_x:
        domain_x = f_x[f]
    if f_y[f] > domain_y:
        domain_y = f_y[f]

# point face normals from cell 0 to cell 1
for i in range(0, num_face):
    c1 = f_c[i, 0]
    c2 = f_c[i, 1]
    if c2 == c1:  # boundary face normal vector points out
        dx = f_x[i] - c_x[c1]
        dy = f_y[i] - c_y[c1]
        n_dot_l = dx * fn_x[i] + dy * fn_y[i]
        if n_dot_l < 0:
            fn_x[i] = - fn_x[i]
            fn_y[i] = - fn_y[i]
            ft_x[i] = - ft_x[i]
            ft_y[i] = - ft_y[i]
    else:  # point from cell 0 to cell 1
        dx = c_x[c2] - c_x[c1]
        dy = c_y[c2] - c_y[c1]
        n_dot_l = dx * fn_x[i] + dy * fn_y[i]
        if n_dot_l < 0:
            fn_x[i] = - fn_x[i]
            fn_y[i] = - fn_y[i]
            ft_x[i] = - ft_x[i]
            ft_y[i] = - ft_y[i]

# delta
for i in range(0, num_face):
    c1 = f_c[i, 0]
    c2 = f_c[i, 1]
    if c2 == c1:  # boundary face
        dx = f_x[i] - c_x[c1]
        dy = f_y[i] - c_y[c1]
        delta[i] = np.abs(dx * fn_x[i] + dy * fn_y[i])
    else:  # interior face
        dx = c_x[c2] - c_x[c1]
        dy = c_y[c2] - c_y[c1]
        delta[i] = np.abs(dx * fn_x[i] + dy * fn_y[i])

# face normal sign modifier, face normals point outward from local cell perspective
for j in range(0, num_cell):
    for i in range(0, 3):
        c1 = f_c[c_f[j, i], 0]
        if j == c1:
            fn_sign[j, i] = 1
        else:
            fn_sign[j, i] = -1

# cell volume
for i in range(0, num_cell):
    v1 = c_v[i, 0]
    v2 = c_v[i, 1]
    v3 = c_v[i, 2]
    dx1 = v_x[v1] - v_x[v3]
    dy1 = v_y[v1] - v_y[v3]
    dx2 = v_x[v2] - v_x[v3]
    dy2 = v_y[v2] - v_y[v3]
    vol[i] = (np.absolute(dx1 * dy2 - dy1 * dx2)) / 2

# boundary vertices flag
for i in range(0, num_face):
    if bf_flag[i] == 1:
        bv_flag[f_v[i, 0]] = 1
        bv_flag[f_v[i, 1]] = 1

# cell-to-face interpolation weights
for i in range(0, num_face):
    c1 = f_c[i, 0]
    c2 = f_c[i, 1]
    dx1 = f_x[i] - c_x[c1]
    dy1 = f_y[i] - c_y[c1]
    dist1 = np.sqrt(dx1 ** 2 + dy1 ** 2)
    dx2 = f_x[i] - c_x[c2]
    dy2 = f_y[i] - c_y[c2]
    dist2 = np.sqrt(dx2 ** 2 + dy2 ** 2)
    wt_c_f[i] = dist2 / (dist1 + dist2)

# cell-to-vertex interpolation weights
for j in range(0, num_cell):
    for i in range(0, 3):
        v = c_v[j, i]
        dist = np.sqrt((v_x[v] - c_x[j]) ** 2 + (v_y[v] - c_y[j]) ** 2)
        wt_c_v[j, i] = 1 / dist
        wtsum_c_v[v] = wtsum_c_v[v] + wt_c_v[j, i]
for j in range(0, num_cell):
    for i in range(0, 3):
        v = c_v[j, i]
        wt_c_v[j, i] = wt_c_v[j, i] / wtsum_c_v[v]
for j in range(0, num_cell):
    for i in range(0, 3):
        v = c_v[j, i]
        wt_check[v] = wt_check[v] + wt_c_v[j, i]
# ---end mesh connectivity & properties---

# ---print mesh info---
if pre_info:
    for i in range(0, num_vertex):
        print("Vertex{}: {} {}".format(i, v_x[i], v_y[i]))
    for i in range(0, num_face):
        print("f_mesh{}: {}".format(i, f_mesh[i]))
    for i in range(0, num_face):
        print("f_v{}: {}".format(i, f_v[i]))
    for i in range(0, num_face):
        print("f_c{}: {}".format(i, f_c[i]))
    for i in range(0, num_cell):
        print("c_f{}: {}".format(i, c_f[i]))
    for i in range(0, num_cell):
        print("c_v{}: {}".format(i, c_v[i]))
    for i in range(0, num_cell):
        print("Cell{}: {} {}".format(i, c_x[i], c_y[i]))
    for i in range(0, num_bface):
        print("bf_f{}: {}".format(i, bf_f[i]))
    for i in range(0, num_face):
        print("f_bf{}: {}".format(i, f_bf[i]))
    for i in range(0, num_face):
        print("bf_flag{}: {}".format(i, bf_flag[i]))
    for i in range(0, num_cell):
        print("vol{}: {}".format(i, vol[i]))
    print("delta:\n{}".format(delta))
    print("area:\n{}".format(area))
    for j in range(0, num_cell):
        print("fn_sign[{}]: {}".format(j, fn_sign[j]))
    print("bv_flag: {}".format(bv_flag))
    print("wt_c_f:\n{}".format(wt_c_f))
    for i in range(0, num_cell):
        print("wt_c_v[{}]: {}".format(i, wt_c_v[i]))
    print("wtsum_c_v:\n{}".format(wtsum_c_v))
    print("wt_check:\n{}".format(wt_check))
print("Number of Vertices: {}".format(num_vertex))
print("Number of Internal Vertices: {}".format(num_ivertex))
print("Number of Boundary Vertices: {}".format(num_bvertex))
print("Number of Faces: {}".format(num_face))
print("Number of Internal Faces: {}".format(num_iface))
print("Number of Boundary Faces: {}".format(num_bface))
print("Number of Cells: {}".format(num_cell))
print("Domain X length: {:.2f}".format(domain_x))
print("Domain Y length: {:.2f}".format(domain_y))
# ---end print mesh info---

# ---plot mesh---
if plot_mesh:
    print("Plotting mesh...")
    fig = plt.figure()

    # plot cell centers, vertices and face midpoints
    if plot_mesh_nodes:
        p_v = plt.scatter(v_x, v_y, c="black", zorder=2, label='Vertex')
        p_c = plt.scatter(c_x, c_y, c="#ffc619", zorder=2, label='Cell Centroid')
        p_f = plt.scatter(f_x, f_y, c="#b1cef2", zorder=2, label='Face Midpoint')

    # plot faces
    for i in range(0, num_face):
        if bf_flag[i] == 1:
            v1 = f_v[i, 0]
            v2 = f_v[i, 1]
            plt.plot([v_x[v1], v_x[v2]], [v_y[v1], v_y[v2]], c="black", zorder=1)
        else:
            v1 = f_v[i, 0]
            v2 = f_v[i, 1]
            plt.plot([v_x[v1], v_x[v2]], [v_y[v1], v_y[v2]], c="#e6e6e6", zorder=1)

    # plot tangent & normal vectors
    if plot_mesh_vectors:
        plt.plot([f_x[0], f_x[0] + fn_x[0] * scale_unit_vectors],
                 [f_y[0], f_y[0] + fn_y[0] * scale_unit_vectors * area_avg],
                 c="#ff85fb", zorder=2, label='Face Normal Vector')
        plt.plot([f_x[0], f_x[0] + ft_x[0] * scale_unit_vectors],
                 [f_y[0], f_y[0] + ft_y[0] * scale_unit_vectors * area_avg],
                 c="#0b6b0b", zorder=2, label='Face Tangent Vector')
        for i in range(1, num_face):
            plt.plot([f_x[i], f_x[i] + fn_x[i] * scale_unit_vectors],
                     [f_y[i], f_y[i] + fn_y[i] * scale_unit_vectors * area_avg],
                     c="#ff85fb", zorder=2)
            plt.plot([f_x[i], f_x[i] + ft_x[i] * scale_unit_vectors],
                     [f_y[i], f_y[i] + ft_y[i] * scale_unit_vectors * area_avg],
                     c="#0b6b0b", zorder=2)

    # plot details
    plt.title("Mesh Display")
    plt.xlabel("X")
    plt.ylabel("Y")
    if plot_mesh_nodes or plot_mesh_vectors:
        plt.legend(loc='lower right', fontsize=8)
    plt.show()
# ---end plot mesh---
# === end mesh setup ===

# === solver ===
# ---solver setup---
# phi initialization
phi_avg = (north_BC + south_BC + east_BC + west_BC) / 4
for i in range(0, num_cell):
    c_phi[i] = phi_avg

# set face boundary conditions
for i in range(0, num_bface):
    if f_y[bf_f[i]] < (area_avg / 1000):
        bphi[i] = south_BC

# link coefficients
for j in range(0, num_cell):
    for i in range(0, 3):
        a = c_f[j, i]
        if bf_flag[a] == 0:  # interior face
            co_O[j] = co_O[j] + area[a] / delta[a]
            co_nb[j, i] = - area[a] / delta[a]
        else:  # boundary face
            co_O[j] = co_O[j] + area[a] / delta[a]
            co_nb[j, i] = 0
            b_rhs[j] = b_rhs[j] + bphi[f_bf[a]] * area[a] / delta[a]

# print pre info
if pre_info:
    print("Initial PHI:\n{}".format(c_phi))
    print("Boundary PHI:\n{}".format(bphi))
    print("co_self:\n{}".format(co_O))
    print("co_nb:\n{}".format(co_nb))
    print("Boundary Right-Hand Side Contribution:\n{}".format(b_rhs))
# ---end solver setup---

# ---outer iteration loop---
n1 = 0  # outer iteration step
c_r2[0] = 1000000
res_target = phi_avg * norm_res_target
print("Solving...")
while c_r2[n1] > res_target:
    # increment outer iteration step
    n1 = n1 + 1

    # compute vertex values
    for i in range(0, num_vertex):
        v_phi[i] = 0
    for j in range(0, num_cell):
        for i in range(0, 3):
            v = c_v[j, i]
            if bv_flag[v] != 1:
                v_phi[v] = v_phi[v] + c_phi[j] * wt_c_v[j, i]

    # set vertex boundary conditions
    for i in range(0, num_vertex):
        if bv_flag[i] == 1:
            if v_y[i] < (area_avg / 1000):
                v_phi[i] = south_BC

    # tangential flux (skew) source
    num_bf_flag = 0
    for j in range(0, num_cell):
        for i in range(0, 3):
            a = c_f[j, i]
            if bf_flag[a] == 1:
                num_bf_flag = num_bf_flag + 1
            else:
                c1 = f_c[a, 0]
                c2 = f_c[a, 1]
                v1 = f_v[a, 0]
                v2 = f_v[a, 1]
                dx = c_x[c2] - c_x[c1]
                dy = c_y[c2] - c_y[c1]
                t_dot_l = ft_x[a] * dx + ft_y[a] * dy
                c_skew[j, i] = t_dot_l * (v_phi[v2] - v_phi[v1]) * fn_sign[j, i] / delta[a]
    for j in range(0, num_cell):
        c_skewsum[j] = 0
        for i in range(0, 3):
            c_skewsum[j] = c_skewsum[j] + c_skew[j, i]

    # Gauss-Seidel iteration
    for n in range(0, num_n2):
        for j in range(0, num_cell):
            sumf = 0
            for i in range(0, 3):  # sum over all neighbour cells
                f = c_f[j, i]
                if fn_sign[j, i] == 1:  # determine neighbour cell index
                    c = f_c[f, 1]
                else:
                    c = f_c[f, 0]
                sumf = sumf + co_nb[j, i] * c_phi[c]
            c_phi[j] = (b_rhs[j] + c_skewsum[j] - sumf) / co_O[j]

    # residual
    for j in range(0, num_cell):
        c_r[j] = 0
        sumf = 0
        for i in range(0, 3):
            f = c_f[j, i]
            if fn_sign[j, i] == 1:
                c = f_c[f, 1]
            else:
                c = f_c[f, 0]
            sumf = sumf + co_nb[j, i] * c_phi[c]
        c_r[j] = b_rhs[j] + c_skewsum[j] - co_O[j] * c_phi[j] - sumf
    c_r2_cur = 0
    for i in range(0, num_cell):
        c_r2_cur = c_r2_cur + c_r[i] ** 2
    c_r2_cur = c_r2_cur / num_cell
    c_r2[n1] = np.sqrt(c_r2_cur)
    c_r2 = np.append(c_r2, 0)

    if ((n1 * num_n2) % n1n2_per_print) == 0:
        print("Iteration{}: R2 = {:.2e}".format(n1, c_r2[n1]))

# ---end outer iteration---
print("Solution Reached")
# === end solver ===

# === post processing ===
# default printing
print("Number of Outer Iterations: {}".format(n1))

# ---print post info---
if post_info:
    print("v_phi:\n{}".format(v_phi))
    print("c_skew:\n{}".format(c_skew))
    print("c_skewsum:\n{}".format(c_skewsum))
    print("c_phi:\n{}".format(c_phi))
    print("c_r:\n{}".format(c_r))
# ---end print post info---

# ---plot convergence---
# plot r2 convergence
if plot_residual:
    c_r2 = np.log10(c_r2[1:(n1 + 1)])  # select data of interest
    n_O = np.arange(n1)  # RMS residual for each timestep
    plt.scatter(n_O, c_r2, c="black")
    plt.title("R2 Convergence")
    plt.xlabel("Outer Iterations")
    plt.ylabel("log(R2)")

# plot residuals
if plot_3D_residual:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_trisurf(c_x, c_y, c_r, color=(0, 0, 0, 0), edgecolor='Gray')
    ax.set_title('Residuals')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Residual')
# ---end plot convergence---

# ---plot solution---
if plot_contour:
    f, ax = plt.subplots(1, 1)
    ax.tricontourf(v_x, v_y, v_phi, contour_res,
                   cmap='viridis')  # choose 20 contour levels, just to show how good its interpolation is
    for i in range(0, num_face):
        if bf_flag[i] == 1:
            v1 = f_v[i, 0]
            v2 = f_v[i, 1]
            ax.plot([v_x[v1], v_x[v2]], [v_y[v1], v_y[v2]], c="black", zorder=1)

# plot vertex solution
if plot_vertex:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_trisurf(v_x, v_y, v_phi, color=(0, 0, 0, 0), edgecolor='Gray')
    ax.set_title('PHI at Vertices')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('v_PHI')

# plot vertex solution
if plot_cell:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_trisurf(c_x, c_y, c_phi, color=(0, 0, 0, 0), edgecolor='Gray')
    ax.set_title('PHI at Cell Centroid')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('c_PHI')
# ---end plot solution---
plt.show()
# === end post processing ===
