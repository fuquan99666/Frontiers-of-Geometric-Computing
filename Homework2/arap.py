

## in this file , we will implement the As-Rigid-As-Possible (ARAP) deformation algorithm for meshes 
## the show demo is deformation_arap_result.obj

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# the most important library for numerical computations and linear algebra
import numpy as np

# We will use scipy's sparse linear solver for the global step of ARAP.
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


@dataclass
class Mesh:
    vertices: np.ndarray  # (N, 3)
    faces: np.ndarray  # (M, 3), 0-based indices


def load_obj(path: str) -> Mesh:
    """Load a triangle OBJ mesh.

    open our deformation.obj file and we will find 
    format is :
    v x y z (x,y,z is vertex position)
    f i j k (i,j,k are vertex indices)
    """
    vertices: List[List[float]] = []  # [[x_1,y_1,z_1], [x_2,y_2,z_2], ...]
    faces: List[List[int]] = []       # [[i_1,j_1,k_1], [i_2,j_2,k_2], ...]

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip() # remove whitespace from the beginning and end of the line 
            if not line or line.startswith("#"): # if the line is empty or starts with #, skip it 
                continue

            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]

                # actually we only have triangles, but for safety, you know.
                if len(parts) != 3:
                    raise ValueError("Only triangle meshes are supported.")

                idx = []

                # OBJ format can be f i j k or f i/m j/n k/p where i,j,k are vertex indices and m,n,p are optional texture/normal indices.
                # though in our deformation.obj, we only have the simple f i j k
                # but to be robust, you know.

                # here is a question, f's indice is 0-based or 1-based ?
                # in OBJ format, 1-based is used.
                for token in parts:
                    vid = token.split("/")[0]
                    idx.append(int(vid) - 1)
                faces.append(idx)

    if not vertices or not faces:
        raise ValueError(f"Invalid OBJ: {path}")

    return Mesh(vertices=np.asarray(vertices, dtype=np.float64), faces=np.asarray(faces, dtype=np.int32))


def save_obj(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:

    # save the deformed mesh to an OBJ file, with 1-based vertex indices in faces 

    with open(path, "w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def _cotangent(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    """Compute cot(theta) = dot(u, v) / ||u x v||."""
    cross_norm = np.linalg.norm(np.cross(u, v))
    if cross_norm < eps:
        # if the angle is very small, actually it means a very big cotangent,
        # but to avoid numerical problems, we treat it as zero.
        return 0.0
    
    return float(np.dot(u, v) / cross_norm)


def build_cotangent_weights(vertices: np.ndarray, faces: np.ndarray) -> Tuple[List[Dict[int, float]], np.ndarray]:

    # build the cotangent weight colection W and the Laplacian matrix L

    n = vertices.shape[0]
    W: List[Dict[int, float]] = [dict() for _ in range(n)]

    for i, j, k in faces:
        pi, pj, pk = vertices[i], vertices[j], vertices[k]

        cot_i = _cotangent(pj - pi, pk - pi)
        cot_j = _cotangent(pi - pj, pk - pj)
        cot_k = _cotangent(pi - pk, pj - pk)

        # Each face contributes 1/2 * cot(opposite angle) to each opposite edge.
        
        w_jk = 0.5 * cot_i
        w_ik = 0.5 * cot_j
        w_ij = 0.5 * cot_k

        for a, b, w in ((j, k, w_jk), (i, k, w_ik), (i, j, w_ij)):
            W[a][b] = W[a].get(b, 0.0) + w
            W[b][a] = W[b].get(a, 0.0) + w
    
    # now , we have the cotangent weights in W, which 
    # records each vertex's neighbors and the corresponding weights.
    # now , let's build the Laplacian matrix L from W.

    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        diag = 0.0
        for j, w in W[i].items():
            
            # W[i] is a dict (neighbor_index -> weight)
            if i == j:
                continue
            L[i, j] = -w  # to build a ax = b system, we use negative value.
            diag += w
        L[i, i] = diag

    # here we can also notice that because one point has only a few neighbors, 
    # so that L is a very sparse matrix, and if we directly create a matrix 
    # that will cause a lot of memory waste !

    return W, L


class ARAP:

    """
    As-Rigid-As-Possible (ARAP) deformation for triangle meshes.
    """

    # we first build a Mesh class by loading the deformation.obj file
    # then we initialize the ARAP class with this mesh 

    # handle_indices is the list of vertex indices that we want to move 
    # handle_positions is the target positions for those handle vertices 

    # question: maybe we can support dynamic interaction by dragging the handle vertices in a GUI?
    # that we will try after finishing the basic target .
    def __init__(self, mesh: Mesh, handle_indices: Sequence[int], handle_positions: np.ndarray):
        self.mesh = mesh
        self.p = mesh.vertices.copy()  # self.p is the original vertex positions
        self.v = mesh.vertices.copy()  # self.v is the current vertex positions, which always in updating 
        self.faces = mesh.faces
        self.n = self.p.shape[0]   # the total number of vertices in the mesh 

        self.handles = np.asarray(handle_indices, dtype=np.int32)
        self.handle_positions = np.asarray(handle_positions, dtype=np.float64)

        if self.handles.ndim != 1:  # handle_indices should be a 1D array of moved vertex
            raise ValueError("handle_indices must be a 1D list/array")
        if self.handle_positions.shape != (self.handles.shape[0], 3):
            raise ValueError("handle_positions must have shape (len(handle_indices), 3)")

        self.W, self.L = build_cotangent_weights(self.p, self.faces)
        self._prepare_system()

    def _prepare_system(self) -> None:

        # handle_indices are the vertices that we want to move directly
        # and the left vertices are free vertices that we will optimize iterably.

        all_idx = np.arange(self.n, dtype=np.int32)
        mask = np.ones(self.n, dtype=bool)
        mask[self.handles] = False
        self.free = all_idx[mask]  # the free vertices are those we should optimize iteratively.

        if self.free.size == 0:
            raise ValueError("All vertices are constrained; no free vertices to optimize.")

        # here we use np.ix_ to extract the submatrices of L 
        # why we differentiate free and handle matrix ?
        # because the postions of handle vertices are fixed so that ...
        
        self.L_ff = self.L[np.ix_(self.free, self.free)]
        self.L_fh = self.L[np.ix_(self.free, self.handles)]

        # because L is a sparse matrix, so we can use csr_matrix to compress it and save memory.
        # csr_matrix use data[],indices[],indptr[] to store the non-zero elements 
        self.L_ff_sparse = csr_matrix(self.L_ff)  

    # first let's claim the logic of the local and global step of ARAP
    # in the local step, we compute the optimal rotation for each vertex's neighborhood
    # in the global step, we solve a linear system to update the vertex positions based on the rotations computed in the local step and the cotangent weights.
    def _local_step_rotations(self) -> np.ndarray:
        
        # init the rotation matrices for each vertex
        R = np.zeros((self.n, 3, 3), dtype=np.float64)

        for i in range(self.n):
            S = np.zeros((3, 3), dtype=np.float64)
            for j, w in self.W[i].items():
                pij = self.p[i] - self.p[j]  # original edge vector 
                vij = self.v[i] - self.v[j]  # deformed edge vector
                S += w * np.outer(vij, pij)  

            U, _, Vt = np.linalg.svd(S)
            Ri = U @ Vt  # actually here and S += , we all make mistake by A and A^T, but it happens twice so that is right .

            # Remove reflections to keep a proper rotation.
            # because in the O(3) group, we have both rotations and reflectiions, 
            # which can be judged by the det, if det < 0, we just filp the sign of it .
            if np.linalg.det(Ri) < 0:
                U[:, -1] *= -1.0
                Ri = U @ Vt
            R[i] = Ri

        return R

    def _global_step(self, R: np.ndarray) -> None:
        
        b = np.zeros((self.n, 3), dtype=np.float64)

        for i in range(self.n):
            rhs = np.zeros(3, dtype=np.float64)
            for j, w in self.W[i].items():
                pij = self.p[i] - self.p[j]

                # edge i_j has two rotation R[i] and R[j]
                # we use the average of them to contribute

                rhs += 0.5 * w * (R[i] + R[j]) @ pij  
            b[i] = rhs

        b_free = b[self.free] - self.L_fh @ self.handle_positions

        x_free = np.zeros((self.free.shape[0], 3), dtype=np.float64)

        for d in range(3):  # x, y, z
            x_free[:, d] = spsolve(self.L_ff_sparse, b_free[:, d])
        
        self.v[self.free] = x_free
        self.v[self.handles] = self.handle_positions

    # one step of the ARAP algorithm
    def update(self) -> None:
        R = self._local_step_rotations()
        self._global_step(R)

    # apply the deformation to the mesh
    def apply(self, iterations: int = 10) -> np.ndarray:
        # Ensure constraints are set from the first iteration.
        self.v[self.handles] = self.handle_positions
        for _ in range(iterations):
            self.update()
        return self.v


def default_handles(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create a stable default deformation setup for deformation.obj.

    Strategy:
    - bottom 8% (by y) are fixed.
    - top 12% (by y) are moved with larger translation and twist.

    This setup intentionally exaggerates the deformation so the ARAP effect
    is easy to observe in the demo.
    """
    y = vertices[:, 1]
    y_low = np.percentile(y, 13.0)
    y_high = np.percentile(y, 80.0)

    fixed_idx = np.where(y <= y_low)[0]
    move_idx = np.where(y >= y_high)[0]

    handles = np.concatenate([fixed_idx, move_idx])
    targets = vertices[handles].copy()

    # Move upper handle vertices to produce strong visible deformation.
    move_mask = np.isin(handles, move_idx)
    bbox = vertices.max(axis=0) - vertices.min(axis=0)

    # 1) Larger global translation for the top region.
    # targets[move_mask, 0] += 0.70 * bbox[0]
    targets[move_mask, 1] += 0.28 * bbox[1]
    targets[move_mask, 2] += 0.35 * bbox[2]

    # # 2) Add a twist around the Y axis centered at the mesh center.
    # center = vertices.mean(axis=0)
    # top_points = targets[move_mask]

    # # Twist angle grows from 0 to max_angle across the selected top handles.
    # y_top = top_points[:, 1]
    # y_min, y_max = np.min(y_top), np.max(y_top)
    # denom = max(y_max - y_min, 1e-8)
    # t = (y_top - y_min) / denom
    # max_angle = np.deg2rad(65.0)
    # angles = max_angle * t

    # rel = top_points - center
    # x, z = rel[:, 0], rel[:, 2]
    # cos_a, sin_a = np.cos(angles), np.sin(angles)
    # x_new = cos_a * x - sin_a * z
    # z_new = sin_a * x + cos_a * z
    # rel[:, 0] = x_new
    # rel[:, 2] = z_new
    # targets[move_mask] = center + rel

    return handles, targets


def show_result(vertices_before: np.ndarray, vertices_after: np.ndarray, faces: np.ndarray) -> None:
    """Visualize original/deformed mesh side by side."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        print("matplotlib is not available; skip visualization.")
        return

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_trisurf(
        vertices_before[:, 0],
        vertices_before[:, 1],
        vertices_before[:, 2],
        triangles=faces,
        color="#9dc3e6",
        linewidth=0.1,
        edgecolor="#4f6f8f",
        alpha=0.9,
    )
    ax1.set_title("Original")
    ax1.set_axis_off()

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_trisurf(
        vertices_after[:, 0],
        vertices_after[:, 1],
        vertices_after[:, 2],
        triangles=faces,
        color="#f4b183",
        linewidth=0.1,
        edgecolor="#8a4f2c",
        alpha=0.9,
    )
    ax2.set_title("ARAP Deformed")
    ax2.set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    input_path = "deformation.obj"
    output_path = "deformation_arap_result.obj"

    mesh = load_obj(input_path)
    handles, targets = default_handles(mesh.vertices)

    solver = ARAP(mesh, handles, targets)
    # More iterations make the strong handle deformation converge better.
    deformed_vertices = solver.apply(iterations=80)

    save_obj(output_path, deformed_vertices, mesh.faces)
    print(f"ARAP deformation finished. Saved to: {output_path}")
    print(f"Vertex count: {mesh.vertices.shape[0]}, Face count: {mesh.faces.shape[0]}, Handle count: {handles.shape[0]}")

    show_result(mesh.vertices, deformed_vertices, mesh.faces)