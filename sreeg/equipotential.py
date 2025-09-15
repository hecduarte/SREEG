import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.spatial import ConvexHull, cKDTree
from matplotlib.patches import Circle, Arc

def plot_electrode_heatmap(positions, values, radius=0.1, cmap="RdBu_r"):
    """
    Plot electrode values on a circular scalp outline (top view).

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Cartesian coordinates of electrodes.
    values : ndarray, shape (N,)
        Electric potential values at electrodes.
    radius : float
        Radius of the circular outline representing the scalp.
    cmap : str
        Colormap to use for electrode values.
    """
    # Proyectar a plano XY (vista superior)
    xy = positions[:, :2]

    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap,
                    s=120, edgecolors="k")

    # Dibujar contorno circular del cráneo
    circle = plt.Circle((0, 0), radius, color="k", fill=False, linewidth=1.5)
    ax.add_artist(circle)

    ax.set_aspect("equal")
    ax.axis("off")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="Potential")

    plt.show()


def plot_equipotential_map(positions, values, radius=0.1, cmap="RdBu_r", subdiv=4):
    """
    Equipotential map over an icosphere (top view).
    Interpolation: linear in XY (no angular seams).
    Returns mesh + per-triangle potentials for downstream analysis.
    """

    # --- Icosahedron (unit sphere) ---
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1)[:, None]

    faces = np.array([
        [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
        [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1],
    ], dtype=int)

    # --- Subdivision (4x per level) ---
    def midpoint(a, b, cache, verts_list):
        key = (a, b) if a < b else (b, a)
        j = cache.get(key)
        if j is not None:
            return j
        v = (verts[a] + verts[b]) / 2.0
        v /= np.linalg.norm(v)
        j = len(verts_list)
        verts_list.append(v)
        cache[key] = j
        return j

    for _ in range(subdiv):
        cache, verts_list, new_faces = {}, verts.tolist(), []
        for a, b, c in faces:
            ab = midpoint(a, b, cache, verts_list)
            bc = midpoint(b, c, cache, verts_list)
            ca = midpoint(c, a, cache, verts_list)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = np.array(new_faces, dtype=int)
        verts = np.array(verts_list)
        verts /= np.linalg.norm(verts, axis=1)[:, None]  # keep unit sphere

    # --- Linear interpolation in XY (no seams) ---
    pos_unit = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    elec_xy  = pos_unit[:, :2]
    verts_xy = verts[:, :2]

    interp = LinearNDInterpolator(elec_xy, values)
    vert_values = interp(verts_xy)

    # Nearest fallback only for vertices outside the convex hull (keeps interior untouched)
    if np.isnan(vert_values).any():
        tree = cKDTree(elec_xy)
        nan_mask = np.isnan(vert_values)
        _, nn_idx = tree.query(verts_xy[nan_mask], k=1)
        vert_values[nan_mask] = values[nn_idx]
    # --- Local smoothing for filled vertices (preserve continuity) ---
    if np.any(nan_mask):
        neighbor_map = [[] for _ in range(len(verts_xy))]
        for a, b, c in faces:
            neighbor_map[a].extend([b, c])
            neighbor_map[b].extend([a, c])
            neighbor_map[c].extend([a, b])
        for v_idx in np.where(nan_mask)[0]:
            neigh = neighbor_map[v_idx]
            if neigh:
                vert_values[v_idx] = np.mean([vert_values[j] for j in neigh if np.isfinite(vert_values[j])])

    # --- Per-triangle potential matrix ---
    # shape (n_faces, 3): vertex potentials for each triangle (consistent with 'faces')
    tri_values = vert_values[faces]
    # Optional convenience: triangle-wise mean (scalar field per face)
    tri_mean = tri_values.mean(axis=1)

    # --- Rendering (top view) ---
    Xg, Yg = verts[:, 0], verts[:, 1]
    fig, ax = plt.subplots(figsize=(5, 5))
    from matplotlib.colors import Normalize
    Vmax = float(np.nanpercentile(np.abs(vert_values), 95))  # robust normalization
    im = ax.tripcolor(Xg, Yg, faces, vert_values, cmap=cmap,
                      norm=Normalize(vmin=-Vmax, vmax=+Vmax), shading="gouraud")

    # --- Head outline (MNE-like) ---
    r = float(np.max(np.hypot(verts[:,0], verts[:,1])))  # ~1.0
    draw_mne_head(ax, r, color="k", lw=1.1, alpha=0.7)

    # --- electrodes (MNE-like dots) ---
    # elec_xy ya lo tienes: pos_unit[:, :2]
    draw_electrodes(ax, elec_xy, r, size=3, color="k", alpha=0.5)#color="#666"

    ax.set_aspect("equal")
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Potential")
    
    # --- Distance-weighted averaging of electrode values onto mesh vertices ---
    # Build k-d tree on electrode XY (neighbor queries use electrode positions).
    tree = cKDTree(elec_xy)
    
    # Number of nearest electrodes considered per vertex (adaptive to NElec).
    NElec = len(values)
    k = int(4 + 128/NElec - (NElec // 128))
    
    # Query k-NN electrodes for all vertices at once.
    dists, nbr_idx = tree.query(verts_xy, k=k)
    
    # Ensure 2D shapes when k == 1.
    if k == 1:
        dists = dists[:, None]
        nbr_idx = nbr_idx[:, None]
    
    # Inverse-distance weights with safe epsilon.
    eps = 1e-6
    w = 1.0 / np.maximum(dists, eps)  # shape (M, k)
    
    # Gather neighbor electrode values and compute weighted average per vertex.
    vals = values[nbr_idx]                              # shape (M, k)
    vert_values = np.sum(vals * w, axis=1) / np.sum(w, axis=1)
    
    # Keep downstream per-triangle matrices consistent with updated vertex field.
    tri_values = vert_values[faces]
    tri_mean = tri_values.mean(axis=1)

    # --- Equipotential curves (continuous paths only) ---
    res = {
        "verts": verts,
        "faces": faces,
        "verts_xy": verts_xy,
        "vertex_values": vert_values,
        "triangle_values": tri_values if 'tri_values' in locals() else vert_values[faces]
    }
    eq = equipotentials(res, n_div=4)

    # style: solid for >0, dashed for <0, thicker for 0
    for L in eq["levels"]:
        lines = eq["polylines"][L]
        if not lines:
            continue
        ls = '-' if L > 0 else (':' if L < 0 else '-')
        lw = 1.1 if L != 0 else 1.1
        for poly in lines:
            ax.plot(poly[:, 0], poly[:, 1], ls=ls, lw=lw, color='k', alpha=0.1)

    #plt.show()

    # Return mesh + values for gradients/contours downstream
    return {
        "verts": verts,                 # (M, 3) unit-sphere vertices
        "faces": faces,                 # (F, 3) vertex indices per triangle
        "verts_xy": verts_xy,           # (M, 2) projected XY (for 2D ops)
        "vertex_values": vert_values,   # (M,) potential at each vertex
        "triangle_values": tri_values,  # (F, 3) per-triangle vertex potentials
        "activity_matrix": tri_values,  # alias: activity matrix
        "triangle_mean": tri_mean       # (F,) scalar per triangle (avg)
    }

def equipotentials(mesh, n_div=4, tol=1e-8, smooth_iters=0, resample_ppu=40, min_len=0.05):
    """
    Fast isolines (equipotentials) extraction.
    - Vectorized edge intersections per level (no Python loop over triangles).
    - Returns: {"levels": levels, "polylines": {level: [polyline_xy, ...]}}
    """

    verts_xy = mesh["verts_xy"]          # (M,2)
    faces    = mesh["faces"]             # (F,3)
    v        = mesh["vertex_values"]     # (M,)

    Vmax = float(np.max(np.abs(v)))
    if not np.isfinite(Vmax) or Vmax <= 0:
        return {"levels": [0.0], "polylines": {0.0: []}}

    # Use relative tolerance (scaled with Vmax)
    tol = 1e-6 * Vmax

    # Levels: negatives, zero, positives (4 divisiones por defecto)
    pos = [(i/n_div)*Vmax for i in range(1, n_div+1)]
    neg = [-x for x in pos]
    levels = neg[::-1] + [0.0] + pos

    # Precompute triangle values/coords (usaremos slicing vectorizado)
    tri_vidx = faces                         # (F,3)
    tri_v  = v[tri_vidx]                     # (F,3)
    tri_xy = verts_xy[tri_vidx]              # (F,3,2)

    # Local edges as arrays for vector ops: (0-1, 1-2, 2-0)
    e0 = np.array([0, 1, 2])
    e1 = np.array([1, 2, 0])

    def _edge_points_for_level(L):
        """Vectorized intersection points for all edges in all triangles at level L."""
        s = tri_v - L                        # (F,3)
        va = s[:, e0]                        # (F,3)
        vb = s[:, e1]                        # (F,3)
        pa = tri_xy[:, e0, :]                # (F,3,2)
        pb = tri_xy[:, e1, :]                # (F,3,2)

        # crossing or hits on level
        cross = (va * vb < 0.0) | (np.abs(va) <= tol) | (np.abs(vb) <= tol)  # (F,3)

        # t on edges; safe denom
        denom = vb - va
        mask_zero = np.abs(denom) < tol

        t = np.empty_like(va)
        # normal case
        t[~mask_zero] = -va[~mask_zero] / denom[~mask_zero]
        # degenerate case: va ≈ vb ≈ 0 → whole edge on level, take midpoint
        t[mask_zero] = 0.5

        # exact hits override t
        t = np.where(np.abs(va) <= tol, 0.0, t)
        t = np.where(np.abs(vb) <= tol, 1.0, t)
        t = np.clip(t, 0.0, 1.0)

        pts = pa + t[..., None] * (pb - pa)  # (F,3,2)
        # mask out non-crossing edges → NaN
        mask = cross[..., None]
        pts = np.where(mask, pts, np.nan)

        # keep at most 2 edge-points per triangle (first two finite)
        finite = np.isfinite(pts[..., 0])    # (F,3)
        count  = finite.sum(axis=1)          # (F,)
        keep   = count >= 2
        if not np.any(keep):
            return np.empty((0, 2)), np.empty((0, 2))  # no segments

        # indices of first two finite per triangle
        idx_first = np.argmax(finite, axis=1)                               # (F,)
        # mask out the first to find second
        finite2 = finite.copy()
        finite2[np.arange(finite2.shape[0]), idx_first] = False
        idx_second = np.argmax(finite2, axis=1)                              # (F,)

        A = pts[np.arange(pts.shape[0]), idx_first]                          # (F,2)
        B = pts[np.arange(pts.shape[0]), idx_second]                         # (F,2)

        return A[keep], B[keep]  # segments as endpoints (K,2),(K,2)

    def _chain_segments(A, B):
        """Build polylines from segment endpoints using light hashing."""
        if A.shape[0] == 0:
            return []
        segs = np.stack([A, B], axis=1)   # (K,2,2)

        def key(p):
            return (round(float(p[0]), 7), round(float(p[1]), 7))

        endpoints = {}
        for i, (a, b) in enumerate(segs):
            endpoints.setdefault(key(a), []).append((i, 0))
            endpoints.setdefault(key(b), []).append((i, 1))

        used = np.zeros(len(segs), dtype=bool)
        chains = []

        for i in range(len(segs)):
            if used[i]:
                continue
            a, b = segs[i]; used[i] = True
            chain = [a, b]

            # forward
            cur = key(b)
            while True:
                nxt = None
                for (j, end) in endpoints.get(cur, []):
                    if used[j]: continue
                    p0, p1 = segs[j]
                    cand = p0 if end == 1 else p1
                    nxt = (j, cand); break
                if nxt is None: break
                j, cand = nxt; used[j] = True
                chain.append(cand); cur = key(cand)

            # backward
            cur = key(a)
            while True:
                nxt = None
                for (j, end) in endpoints.get(cur, []):
                    if used[j]: continue
                    p0, p1 = segs[j]
                    cand = p1 if end == 0 else p0
                    nxt = (j, cand); break
                if nxt is None: break
                j, cand = nxt; used[j] = True
                chain = [cand] + chain; cur = key(cand)

            chains.append(np.asarray(chain))

        # filter very short chains (noise)
        if min_len is not None and min_len > 0:
            lens = [np.linalg.norm(np.diff(c, axis=0), axis=1).sum() for c in chains]
            chains = [c for c, L in zip(chains, lens) if L >= min_len]
        return chains

    # (optional) lightweight smoothing + resampling
    def _smooth_resample(poly, iters=0, n_per_unit=40):
        if iters <= 0 or len(poly) < 3:
            return poly
        P = poly.copy()
        for _ in range(iters):
            Q = 0.75 * P[:-1] + 0.25 * P[1:]
            R = 0.25 * P[:-1] + 0.75 * P[1:]
            P = np.vstack([P[0:1], np.column_stack([Q, R]).reshape(-1, 2), P[-1:]])
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        L = np.concatenate([[0.0], np.cumsum(seg)])
        total = L[-1] if L[-1] > 0 else 1.0
        m = max(12, int(np.ceil(total * n_per_unit)))
        t = np.linspace(0, total, m)
        x = np.interp(t, L, P[:, 0]); y = np.interp(t, L, P[:, 1])
        return np.column_stack([x, y])

    polylines = {}
    for L in levels:
        A, B = _edge_points_for_level(L)        # vectorizado
        chains = _chain_segments(A, B)          # encadenado ligero
        if smooth_iters > 0 and chains:
            chains = [_smooth_resample(c, iters=smooth_iters, n_per_unit=resample_ppu) for c in chains]
        polylines[L] = chains

    return {"levels": levels, "polylines": polylines}

from matplotlib.patches import Circle, Arc

def draw_mne_head(ax, r, color="k", lw=1.1, alpha=0.7):
    """
    MNE-like outline:
      - scalp: circle of radius r
      - nose: skinny isosceles triangle at the top
      - ears: half-ellipses entirely outside the scalp (do not intrude)
    """
    # --- scalp ---
    ax.add_patch(Circle((0, 0), r, fill=False,
                        edgecolor=color, lw=lw, alpha=alpha, zorder=10))

    # --- nose (narrow) ---
    nose_half_deg = 12.0                         # half opening angle (skinny)
    a1, a2 = np.deg2rad(90 - nose_half_deg), np.deg2rad(90 + nose_half_deg)
    bx1, by1 = r*np.cos(a1), r*np.sin(a1)       # base on the circle
    bx2, by2 = r*np.cos(a2), r*np.sin(a2)
    tip = (0.0, r*1.16)                         # tip slightly outside
    ax.plot([bx1, tip[0], bx2], [by1, tip[1], by2],
            color=color, lw=lw, alpha=alpha, solid_capstyle='round', zorder=11)

    # --- ears (half-ellipses fully outside) ---
    # Choose semi-axes; ensure center_x = r + margin + a so inner edge > r
    a = 0.12*r          # semi-axis (x)  → ear "width"
    b = 0.22*r           # semi-axis (y)  → ear "height"
    margin = -0.14*r      # clearance so stroke never crosses inside the scalp
    cx = r + margin + a  # ear center x-position

    ear_w, ear_h = 2*a, 2*b
    right = Arc((+cx, 0.0), ear_w, ear_h, angle=0,
                theta1=-90, theta2=90, edgecolor=color, lw=lw, alpha=alpha, zorder=10)
    left  = Arc((-cx, 0.0), ear_w, ear_h, angle=0,
                theta1=90, theta2=270, edgecolor=color, lw=lw, alpha=alpha, zorder=10)
    ax.add_patch(left); ax.add_patch(right)

def draw_electrodes(ax, elec_xy, r, size=10, color="#444", alpha=0.8):
    """
    Plot electrode locations as small dots on top of the head.
    - elec_xy: (N,2) in unit XY (already from positions normalized).
    - r: scalp radius (≈1.0).
    """
    
    # keep only points inside scalp (numerical margin)
    m = np.hypot(elec_xy[:,0], elec_xy[:,1]) <= (r * 0.9985)
    exy = elec_xy[m]

    sc = ax.scatter(exy[:,0], exy[:,1],
                    s=size, facecolors=color, edgecolors=color,
                    linewidths=0.1, alpha=alpha, zorder=20)
    return sc

