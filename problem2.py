from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import trimesh

Array = jax.Array
GridShape = Tuple[int, int, int]


Array = jax.Array

class DenseDiffMCResult(NamedTuple):
    """Static-shape dense DiffMC output before compaction.

    Attributes:
        vertices_dense:
            Shape `[num_cells, 3, 3]`. For each cell, stores the candidate
            vertices on the cell's +x, +y, and +z edges.
        vertex_valid_mask:
            Shape `[num_cells, 3]`. `True` when the corresponding dense vertex
            is active and belongs to the extracted iso-surface.
        faces_dense:
            Shape `[num_cells, 5, 3]`. Up to five triangle faces per cell,
            stored as indices into the flattened dense vertex buffer.
        face_valid_mask:
            Shape `[num_cells, 5]`. `True` when the corresponding face slot is
            active.
        cube_codes:
            Shape `[num_cells]`. Marching Cubes case id in `[0, 255]`.
    """

    vertices_dense: Array
    vertex_valid_mask: Array
    faces_dense: Array
    face_valid_mask: Array
    cube_codes: Array

class CompactDiffMCResult(NamedTuple):
    """Static-shape compacted DiffMC mesh."""

    vertices: Array
    faces: Array
    num_vertices: Array
    num_faces: Array


#Useful lookup tables.  Sorry it's so ugly!
MC_CORNERS = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [1, 0, 1],
                       [1, 1, 1],
                       [0, 1, 1]], dtype=np.int32)

EDGE_VERTEX_LOOKUP = np.array([[0, 0, 0, 0],
                               [1, 0, 0, 2],
                               [0, 0, 1, 0],
                               [0, 0, 0, 2],
                               [0, 1, 0, 0],
                               [1, 1, 0, 2],
                               [0, 1, 1, 0],
                               [0, 1, 0, 2],
                               [0, 0, 0, 1],
                               [1, 0, 0, 1],
                               [1, 0, 1, 1],
                               [0, 0, 1, 1]], dtype=np.int32)

_EDGE_AXIS_OFFSETS = np.eye(3, dtype=np.int32)

_corner_coordinate_to_id = {
    tuple(corner.tolist()): int(index) for index, corner in enumerate(MC_CORNERS)
}

EDGE_CORNER_PAIRS = np.asarray(
    [
        [
            _corner_coordinate_to_id[tuple(edge[:3].tolist())],
            _corner_coordinate_to_id[
                tuple((edge[:3] + _EDGE_AXIS_OFFSETS[int(edge[3])]).tolist())
            ],
        ]
        for edge in EDGE_VERTEX_LOOKUP
    ],
    dtype=np.int32,
)

_CANONICAL_CASE_MASKS = np.asarray(
    [0, 1, 3, 5, 7, 15, 20, 21, 23, 26, 27, 29, 30, 45, 60, 61, 90, 91, 95, 150, 165],
    dtype=np.int32,
)

_CANONICAL_TRIANGLE_TABLE = np.full((21, 5, 3), -1, dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[0, :0] = np.empty((0, 3), dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[1, :1] = np.array([[0, 3, 8]], dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[2, :2] = np.array([[9, 3, 8], [1, 3, 9]], dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[3, :2] = np.array([[0, 3, 8], [9, 4, 5]], dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[4, :3] = np.array([[4, 3, 8], [4, 5, 3], [5, 1, 3]], dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[5, :2] = np.array([[1, 3, 5], [5, 3, 7]], dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[6, :2] = np.array([[9, 4, 5], [3, 2, 11]], dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[7, :3] = np.array([[8, 2, 11], [8, 0, 2], [9, 4, 5]], dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[8, :4] = np.array([[4, 5, 1], [4, 1, 11], [4, 11, 8], [11, 1, 2]], dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[9, :3] = np.array([[1, 0, 9], [3, 2, 11], [4, 8, 7]], dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[10, :4] = np.array([[2, 11, 7], [1, 2, 7], [1, 7, 4], [1, 4, 9]], dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[11, :4] = np.array([[9, 7, 5], [9, 2, 7], [9, 0, 2], [11, 7, 2]], dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[12, :4] = np.array([[2, 11, 3], [1, 0, 7], [1, 7, 5], [7, 0, 8]], dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[13, :4] = np.array([[2, 1, 10], [0, 3, 9], [3, 5, 9], [3, 7, 5]], dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[14, :4] = np.array([[1, 10, 3], [3, 10, 11], [5, 9, 8], [5, 8, 7]], dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[15, :5] = np.array([[10, 11, 0], [10, 0, 1], [11, 7, 0], [9, 0, 5], [7, 5, 0]], dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[16, :4] = np.array([[0, 9, 1], [2, 11, 3], [4, 8, 7], [10, 5, 6]], dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[17, :5] = np.array([[1, 4, 9], [1, 7, 4], [1, 2, 7], [11, 7, 2], [10, 5, 6]], dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[18, :4] = np.array([[6, 10, 1], [6, 1, 7], [2, 11, 1], [11, 7, 1]], dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[19, :4] = np.array([[2, 7, 3], [2, 6, 7], [0, 4, 1], [4, 5, 1]], dtype=np.int32)
_CANONICAL_TRIANGLE_TABLE[20, :4] = np.array([[6, 7, 11], [9, 4, 5], [0, 3, 8], [2, 1, 10]], dtype=np.int32)


_CASE_TRANSFORM_CODES = np.asarray(
    [
        0, 225, 97, 226, 545, 227, 834, 228, 609, 610, 675, 612, 34, 36, 676, 2245,
        193, 1250, 355, 356, 230, 231, 359, 232, 1379, 1380, 233, 234, 1447, 235, 236, 2180,
        1057, 1059, 1058, 1060, 835, 361, 836, 362, 838, 1255, 679, 3147, 839, 237, 840, 2756,
        418, 1252, 420, 3077, 903, 1069, 2859, 2308, 1383, 1256, 365, 3204, 238, 239, 367, 2306,
        1025, 1062, 1475, 1479, 1282, 39, 1284, 2187, 451, 455, 841, 685, 452, 40, 842, 2116,
        515, 711, 1065, 1068, 519, 1070, 1485, 1487, 1097, 812, 240, 241, 1100, 1455, 849, 242,
        898, 1063, 1476, 1064, 900, 845, 837, 3492, 263, 846, 844, 687, 843, 847, 2852, 2754,
        516, 1067, 1066, 3140, 904, 1071, 3044, 3490, 140, 719, 1073, 1074, 271, 2662, 850, 2625,
        577, 995, 614, 999, 1155, 457, 1159, 45, 1442, 996, 615, 2952, 1092, 3114, 3115, 2564,
        706, 804, 423, 2891, 1095, 1005, 243, 1007, 1444, 2885, 1004, 2948, 3112, 3524, 623, 2946,
        67, 1385, 71, 1261, 521, 244, 909, 369, 135, 1388, 851, 1263, 1165, 465, 1167, 690,
        68, 2890, 2088, 2500, 524, 1393, 911, 370, 139, 3332, 1391, 3330, 143, 1394, 3110, 3073,
        258, 807, 1287, 1075, 1156, 460, 3304, 47, 260, 811, 1164, 463, 1029, 2468, 3300, 2466,
        708, 2888, 76, 431, 1099, 1103, 527, 2886, 2410, 2884, 1105, 1010, 3108, 3106, 1170, 3105,
        132, 77, 2283, 79, 2282, 529, 3428, 1490, 2280, 815, 1295, 2278, 2404, 466, 3298, 2241,
        197, 2724, 2084, 2082, 2660, 530, 2658, 2657, 2276, 2882, 82, 2593, 2274, 2145, 2273, 2048,
    ],
    dtype=np.uint16,
)

def transform_case_mask(mask: int, vertex_permutation: np.ndarray) -> int:
    """Apply a cube symmetry to a case bit-mask."""
    transformed_mask = 0
    for source_vertex in range(8):
        if (mask >> source_vertex) & 1:
            transformed_mask |= 1 << int(vertex_permutation[source_vertex])
    return int(transformed_mask)

def _generate_cube_symmetry_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the 48 signed-permutation symmetries of the unit cube.

    Returns:
        A tuple of:
        - vertex permutations with shape `[48, 8]`,
        - edge permutations with shape `[48, 12]`,
        - determinants with shape `[48]`, each in `{+1, -1}`.
    """
    cube_center = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    corner_coordinate_to_id = {
        tuple(corner.tolist()): int(index) for index, corner in enumerate(MC_CORNERS)
    }
    edge_pair_to_id = {
        tuple(sorted((int(edge[0]), int(edge[1])))): int(edge_id)
        for edge_id, edge in enumerate(EDGE_CORNER_PAIRS)
    }

    vertex_permutations: list[np.ndarray] = []
    edge_permutations: list[np.ndarray] = []
    determinants: list[int] = []

    for axis_permutation in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        permutation_matrix = np.eye(3, dtype=np.int32)[:, axis_permutation]
        for axis_signs in (
            (-1, -1, -1),
            (-1, -1, 1),
            (-1, 1, -1),
            (-1, 1, 1),
            (1, -1, -1),
            (1, -1, 1),
            (1, 1, -1),
            (1, 1, 1),
        ):
            signed_permutation = permutation_matrix * np.asarray(axis_signs, dtype=np.int32)
            determinant = int(round(np.linalg.det(signed_permutation)))
            transformed_corners = (
                (MC_CORNERS.astype(np.float64) - cube_center) @ signed_permutation.T
                + cube_center
            ).round().astype(np.int32)

            if {
                tuple(corner.tolist()) for corner in transformed_corners
            } != {tuple(corner.tolist()) for corner in MC_CORNERS}:
                continue

            vertex_permutation = np.asarray(
                [
                    corner_coordinate_to_id[tuple(corner.tolist())]
                    for corner in transformed_corners
                ],
                dtype=np.int32,
            )

            edge_permutation = np.asarray(
                [
                    edge_pair_to_id[
                        tuple(
                            sorted(
                                (
                                    int(vertex_permutation[int(edge[0])]),
                                    int(vertex_permutation[int(edge[1])]),
                                )
                            )
                        )
                    ]
                    for edge in EDGE_CORNER_PAIRS
                ],
                dtype=np.int32,
            )

            vertex_permutations.append(vertex_permutation)
            edge_permutations.append(edge_permutation)
            determinants.append(determinant)

    vertex_permutations_array = np.asarray(vertex_permutations, dtype=np.int32)
    edge_permutations_array = np.asarray(edge_permutations, dtype=np.int32)
    determinants_array = np.asarray(determinants, dtype=np.int32)

    if vertex_permutations_array.shape != (48, 8):
        raise RuntimeError(
            "Expected to generate exactly 48 cube symmetries, "
            f"got vertex permutations with shape {vertex_permutations_array.shape}."
        )
    if edge_permutations_array.shape != (48, 12):
        raise RuntimeError(
            "Expected edge permutations with shape `(48, 12)`, "
            f"got {edge_permutations_array.shape}."
        )
    if determinants_array.shape != (48,):
        raise RuntimeError(
            "Expected determinant array with shape `(48,)`, "
            f"got {determinants_array.shape}."
        )

    return vertex_permutations_array, edge_permutations_array, determinants_array

_CUBE_VERTEX_PERMUTATIONS, _CUBE_EDGE_PERMUTATIONS, _CUBE_SYMMETRY_DETERMINANTS = _generate_cube_symmetry_data()

def unpack_case_transform_code(code: int) -> tuple[int, int, bool]:
    """Unpack one tiny case-transform record."""
    canonical_case_index = int(code & 0x1F)
    symmetry_index = int((code >> 5) & 0x3F)
    complemented = bool((code >> 11) & 0x1)
    return canonical_case_index, symmetry_index, complemented

def transform_canonical_triangles(
    canonical_triangles: np.ndarray,
    edge_permutation: np.ndarray,
    symmetry_determinant: int,
    complemented: bool,
) -> np.ndarray:
    """Transform a canonical case through a cube symmetry and optional complement."""
    transformed = np.array(canonical_triangles, copy=True)
    valid_triangle_mask = np.all(transformed >= 0, axis=-1)
    valid_triangles = transformed[valid_triangle_mask]

    flip_winding = bool(symmetry_determinant < 0) ^ bool(complemented)
    if flip_winding:
        valid_triangles = valid_triangles[:, [0, 2, 1]]

    transformed_valid_triangles = edge_permutation[valid_triangles]
    transformed = np.full_like(canonical_triangles, fill_value=-1)
    transformed[: transformed_valid_triangles.shape[0]] = transformed_valid_triangles
    return transformed

def build_triangle_table_from_canonical_cases() -> np.ndarray:
    """Build the full 256-case MC triangle table from canonical cases + symmetries.

    This keeps the source code compact and readable while reproducing the exact
    standard MC / DISO connectivity used by the rest of the DiffMC code.
    """
    triangle_table = np.full((256, 5, 3), -1, dtype=np.int32)

    for case_code in range(256):
        canonical_case_index, symmetry_index, complemented = unpack_case_transform_code(
            int(_CASE_TRANSFORM_CODES[case_code])
        )

        canonical_case_mask = int(_CANONICAL_CASE_MASKS[canonical_case_index])
        transformed_case_mask = transform_case_mask(
            canonical_case_mask,
            _CUBE_VERTEX_PERMUTATIONS[symmetry_index],
        )
        if complemented:
            transformed_case_mask ^= 0xFF
        if transformed_case_mask != case_code:
            raise RuntimeError(
                "Packed case-transform metadata is inconsistent: "
                f"expected case {case_code}, got {transformed_case_mask}."
            )

        triangle_table[case_code] = transform_canonical_triangles(
            canonical_triangles=_CANONICAL_TRIANGLE_TABLE[canonical_case_index],
            edge_permutation=_CUBE_EDGE_PERMUTATIONS[symmetry_index],
            symmetry_determinant=int(_CUBE_SYMMETRY_DETERMINANTS[symmetry_index]),
            complemented=complemented,
        )

    return triangle_table

TRIANGLE_TABLE = build_triangle_table_from_canonical_cases()

MAX_TRIANGLES_PER_CELL = int(TRIANGLE_TABLE.shape[1])

INVALID_EDGE_ID = EDGE_VERTEX_LOOKUP.shape[0]

#End of lookup table generation

Array = jax.Array

GridShape = Tuple[int, int, int]

def pad_field(field: Array, isovalue: float, pad_value: float = 1.0) -> Array:
    """Pad a scalar field with a constant value strictly outside the iso-surface."""
    return jnp.pad(
        field,
        ((1, 1), (1, 1), (1, 1)),
        mode="constant",
        constant_values=isovalue + pad_value,
    )

def pad_deformation(deformation: Array | None) -> Array | None:
    """Pad a deformation field with zeros."""
    if deformation is None:
        return None
    return jnp.pad(
        deformation,
        ((1, 1), (1, 1), (1, 1), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )

def make_regular_grid(
    shape: GridShape,
    bounds_min: Array | tuple[float, float, float],
    bounds_max: Array | tuple[float, float, float],
    dtype: jnp.dtype = jnp.float32,
) -> Array:
    """Create a dense regular 3D grid of sample points."""
    bounds_min_arr = jnp.asarray(bounds_min, dtype=dtype)
    bounds_max_arr = jnp.asarray(bounds_max, dtype=dtype)

    xs = jnp.linspace(bounds_min_arr[0], bounds_max_arr[0], shape[0], dtype=dtype)
    ys = jnp.linspace(bounds_min_arr[1], bounds_max_arr[1], shape[1], dtype=dtype)
    zs = jnp.linspace(bounds_min_arr[2], bounds_max_arr[2], shape[2], dtype=dtype)
    return jnp.stack(jnp.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)

def make_cell_coordinates(cell_shape: GridShape) -> Array:
    """Return the base coordinate of each cell in flattened x-major order."""
    xs = jnp.arange(cell_shape[0], dtype=jnp.int32)
    ys = jnp.arange(cell_shape[1], dtype=jnp.int32)
    zs = jnp.arange(cell_shape[2], dtype=jnp.int32)
    return jnp.stack(jnp.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

def linearize_cell_coordinates(cell_coordinates: Array, cell_shape: GridShape) -> Array:
    """Flatten cell coordinates using the same x-major order as the CUDA code."""
    _, cy, cz = (int(cell_shape[0]), int(cell_shape[1]), int(cell_shape[2]))
    return cell_coordinates[..., 2] + cz * (
        cell_coordinates[..., 1] + cy * cell_coordinates[..., 0]
    )

def normalize_vertices(vertices: Array, original_grid_shape: GridShape) -> Array:
    """Map vertices from grid coordinates to `[0, 1]^3`."""
    denominator = jnp.asarray(
        [
            max(original_grid_shape[0] - 1, 1),
            max(original_grid_shape[1] - 1, 1),
            max(original_grid_shape[2] - 1, 1),
        ],
        dtype=vertices.dtype,
    )
    return vertices / denominator[None, :]

def rescale_unit_vertices(
    unit_vertices: Array,
    bounds_min: Array | tuple[float, float, float],
    bounds_max: Array | tuple[float, float, float],
) -> Array:
    """Affine-map vertices from `[0, 1]^3` to an arbitrary axis-aligned box."""
    bounds_min_arr = jnp.asarray(bounds_min, dtype=unit_vertices.dtype)
    bounds_max_arr = jnp.asarray(bounds_max, dtype=unit_vertices.dtype)
    return unit_vertices * (bounds_max_arr - bounds_min_arr)[None, :] + bounds_min_arr[None, :]



_BASE_EDGE_OFFSETS = jnp.eye(3, dtype=jnp.int32)

_EDGE_VERTEX_LOOKUP_PADDED = jnp.concatenate(
    [jnp.asarray(EDGE_VERTEX_LOOKUP, dtype=jnp.int32), jnp.zeros((1, 4), dtype=jnp.int32)],
    axis=0,
)

_MC_CORNERS_JAX = jnp.asarray(MC_CORNERS, dtype=jnp.int32)

_TRIANGLE_TABLE_JAX = jnp.asarray(TRIANGLE_TABLE, dtype=jnp.int32)

def gather_cell_corner_values(field_padded: Array, cell_coordinates: Array) -> Array:
    """Gather the eight corner values of each cell.

    Returns:
        Shape `[num_cells, 8]`.
    """

    def sample_one_corner(corner_offset: Array) -> Array:
        corner_coordinates = cell_coordinates + corner_offset[None, :]
        return field_padded[
            corner_coordinates[:, 0],
            corner_coordinates[:, 1],
            corner_coordinates[:, 2],
        ]

    return jax.vmap(sample_one_corner)(_MC_CORNERS_JAX).T

def compute_cube_codes(corner_values: Array, isovalue: float) -> Array:
    """Compute the 8-bit Marching Cubes case code for each cell.

    Bit i is set when corner i is strictly inside the iso-surface
    (i.e. its value is less than the isovalue).

    Args:
        corner_values: Shape `[num_cells, 8]`, scalar field sampled at the
            eight corners of each cell.
        isovalue: Iso-surface threshold.

    Returns:
        Shape `[num_cells]`, integer codes in `[0, 255]`.
    """
    # inside[i, j] = True when corner j of cell i is inside the surface
    inside = corner_values < isovalue  # [num_cells, 8]
    # Each corner contributes 2^j to the code
    bit_weights = jnp.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=jnp.int32)
    return jnp.sum(inside.astype(jnp.int32) * bit_weights[None, :], axis=-1)

def interpolate_owned_edge_vertex(
    field_padded: Array,
    deformation_padded: Array | None,
    cell_coordinates: Array,
    edge_offset: Array,
    isovalue: float,
) -> tuple[Array, Array]:
    """Interpolate one of the three owned edge vertices per cell.

    Args:
        field_padded:
            Padded scalar field, shape `[nx + 2, ny + 2, nz + 2]`.
        deformation_padded:
            Optional padded deformation field, shape `[nx + 2, ny + 2, nz + 2, 3]`.
        cell_coordinates:
            Flattened cell base coordinates, shape `[num_cells, 3]`.
        edge_offset:
            One of `[1, 0, 0]`, `[0, 1, 0]`, or `[0, 0, 1]`.
        isovalue:
            Iso-surface threshold.

    Returns:
        A tuple `(positions, valid_mask)` with shapes `[num_cells, 3]` and
        `[num_cells]`.
    """

    # Endpoint 0: cell_coordinates (base corner of the edge)
    # Endpoint 1: cell_coordinates + edge_offset (other end of the edge)
    end_coords = cell_coordinates + edge_offset[None, :]  # [num_cells, 3]

    # Sample scalar field at both endpoints
    v0 = field_padded[
        cell_coordinates[:, 0],
        cell_coordinates[:, 1],
        cell_coordinates[:, 2],
    ]  # [num_cells]
    v1 = field_padded[
        end_coords[:, 0],
        end_coords[:, 1],
        end_coords[:, 2],
    ]  # [num_cells]

    # Linear interpolation: find t s.t. v0 + t*(v1-v0) = isovalue
    denom = v1 - v0
    safe_denom = jnp.where(jnp.abs(denom) > 1e-10, denom, jnp.ones_like(denom))
    t = (isovalue - v0) / safe_denom
    t = jnp.clip(t, 0.0, 1.0)  # [num_cells]

    # Interpolated position in grid coordinates
    p0 = cell_coordinates.astype(jnp.float32)  # [num_cells, 3]
    positions = p0 + t[:, None] * edge_offset[None, :].astype(jnp.float32)

    # Optionally add interpolated deformation
    if deformation_padded is not None:
        d0 = deformation_padded[
            cell_coordinates[:, 0],
            cell_coordinates[:, 1],
            cell_coordinates[:, 2],
        ]  # [num_cells, 3]
        d1 = deformation_padded[
            end_coords[:, 0],
            end_coords[:, 1],
            end_coords[:, 2],
        ]  # [num_cells, 3]
        positions = positions + (1.0 - t[:, None]) * d0 + t[:, None] * d1

    # A vertex is valid only when the edge actually crosses the iso-surface
    # (one endpoint inside, one outside)
    valid = (v0 < isovalue) != (v1 < isovalue)  # [num_cells]

    return positions, valid

def build_dense_vertices(
    field_padded: Array,
    deformation_padded: Array | None,
    cell_coordinates: Array,
    isovalue: float,
) -> tuple[Array, Array]:
    """Build the dense owned-edge vertex buffer.

    Returns:
        `vertices_dense` with shape `[num_cells, 3, 3]` and
        `vertex_valid_mask` with shape `[num_cells, 3]`.
    """

    def interpolate_single_owned_edge(edge_offset: Array) -> tuple[Array, Array]:
        return interpolate_owned_edge_vertex(
            field_padded=field_padded,
            deformation_padded=deformation_padded,
            cell_coordinates=cell_coordinates,
            edge_offset=edge_offset,
            isovalue=isovalue,
        )

    vertices_per_edge, valid_per_edge = jax.vmap(interpolate_single_owned_edge)(_BASE_EDGE_OFFSETS)
    return jnp.swapaxes(vertices_per_edge, 0, 1), jnp.swapaxes(valid_per_edge, 0, 1)

def map_one_edge_id_to_dense_vertex_id(
    cell_coordinate: Array,
    edge_id: Array,
    cell_shape: tuple[int, int, int],
) -> Array:
    """Map a Marching Cubes edge id to the corresponding dense owned-vertex id."""
    safe_edge_id = jnp.where(edge_id >= 0, edge_id, INVALID_EDGE_ID)
    edge_lookup = _EDGE_VERTEX_LOOKUP_PADDED[safe_edge_id]
    anchor_cell_coordinate = cell_coordinate + edge_lookup[:3]
    anchor_cell_id = linearize_cell_coordinates(anchor_cell_coordinate, cell_shape)
    dense_vertex_id = 3 * anchor_cell_id + edge_lookup[3]
    return jnp.where(edge_id >= 0, dense_vertex_id, 0)

_map_one_face = jax.vmap(map_one_edge_id_to_dense_vertex_id, in_axes=(None, 0, None))

_map_faces_for_one_cell = jax.vmap(_map_one_face, in_axes=(None, 0, None))

_map_faces_for_all_cells = jax.vmap(_map_faces_for_one_cell, in_axes=(0, 0, None))

def build_dense_faces(
    cube_codes: Array,
    cell_coordinates: Array,
    cell_shape: tuple[int, int, int],
) -> tuple[Array, Array]:
    """Build the dense face buffer with up to five triangles per cell."""
    per_cell_edge_ids = _TRIANGLE_TABLE_JAX[cube_codes]
    faces_dense = _map_faces_for_all_cells(cell_coordinates, per_cell_edge_ids, cell_shape)
    face_valid_mask = jnp.all(per_cell_edge_ids >= 0, axis=-1)
    faces_dense = jnp.where(face_valid_mask[..., None], faces_dense, 0)
    return faces_dense, face_valid_mask

def extract_dense_from_padded_grid(
    field_padded: Array,
    deformation_padded: Array | None,
    isovalue: float = 0.0,
) -> DenseDiffMCResult:
    """Run the dense static-shape DiffMC extraction on a padded grid."""
    cell_shape = (
        field_padded.shape[0] - 1,
        field_padded.shape[1] - 1,
        field_padded.shape[2] - 1,
    )
    cell_coordinates = make_cell_coordinates(cell_shape)
    corner_values = gather_cell_corner_values(field_padded, cell_coordinates)
    cube_codes = compute_cube_codes(corner_values, isovalue)
    vertices_dense, vertex_valid_mask = build_dense_vertices(
        field_padded=field_padded,
        deformation_padded=deformation_padded,
        cell_coordinates=cell_coordinates,
        isovalue=isovalue,
    )
    faces_dense, face_valid_mask = build_dense_faces(
        cube_codes=cube_codes,
        cell_coordinates=cell_coordinates,
        cell_shape=cell_shape,
    )

    return DenseDiffMCResult(
        vertices_dense=vertices_dense,
        vertex_valid_mask=jax.lax.stop_gradient(vertex_valid_mask),
        faces_dense=faces_dense,
        face_valid_mask=jax.lax.stop_gradient(face_valid_mask),
        cube_codes=jax.lax.stop_gradient(cube_codes),
    )

def compact_dense_result(dense_result: DenseDiffMCResult) -> CompactDiffMCResult:
    """Compact the dense static-shape buffers into padded vertex/face arrays."""
    flat_vertices = dense_result.vertices_dense.reshape(-1, 3)
    flat_vertex_valid_mask = dense_result.vertex_valid_mask.reshape(-1)

    vertex_new_ids = (
        jnp.cumsum(flat_vertex_valid_mask.astype(jnp.int32), axis=0, dtype=jnp.int32) - 1
    )
    num_vertices = jnp.sum(flat_vertex_valid_mask.astype(jnp.int32), dtype=jnp.int32)
    max_num_vertices = flat_vertices.shape[0]

    vertex_indices = jnp.nonzero(
        flat_vertex_valid_mask,
        size=max_num_vertices,
        fill_value=0,
    )[0]
    compact_vertices = flat_vertices[vertex_indices]
    compact_vertices = jnp.where(
        (jnp.arange(max_num_vertices, dtype=jnp.int32) < num_vertices)[:, None],
        compact_vertices,
        jnp.zeros_like(compact_vertices),
    )

    faces_in_compact_vertex_space = vertex_new_ids[dense_result.faces_dense]
    flat_face_valid_mask = dense_result.face_valid_mask.reshape(-1)
    faces_in_compact_vertex_space = faces_in_compact_vertex_space.reshape(-1, 3)
    faces_in_compact_vertex_space = jnp.where(
        flat_face_valid_mask[:, None],
        faces_in_compact_vertex_space,
        0,
    )
    num_faces = jnp.sum(flat_face_valid_mask.astype(jnp.int32), dtype=jnp.int32)
    max_num_faces = faces_in_compact_vertex_space.shape[0]

    face_indices = jnp.nonzero(
        flat_face_valid_mask,
        size=max_num_faces,
        fill_value=0,
    )[0]
    compact_faces = faces_in_compact_vertex_space[face_indices]
    compact_faces = jnp.where(
        (jnp.arange(max_num_faces, dtype=jnp.int32) < num_faces)[:, None],
        compact_faces,
        jnp.zeros_like(compact_faces),
    )

    return CompactDiffMCResult(
        vertices=compact_vertices,
        faces=compact_faces.astype(jnp.int32),
        num_vertices=num_vertices,
        num_faces=num_faces,
    )

def extract_dense(
    field: Array,
    deformation: Array | None = None,
    isovalue: float = 0.0,
) -> DenseDiffMCResult:
    """Extract the dense DiffMC buffers from an unpadded grid."""
    return extract_dense_from_padded_grid(
        field_padded=pad_field(field, isovalue=isovalue),
        deformation_padded=pad_deformation(deformation),
        isovalue=isovalue,
    )

def extract_mesh(
    field: Array,
    deformation: Array | None = None,
    isovalue: float = 0.0,
    normalize: bool = True,
) -> CompactDiffMCResult:
    """Extract a compact padded mesh from an unpadded scalar grid.

    This is the main public API. The function is intentionally pure and easy to
    swap out in a notebook or assignment.

    Notes:
        - The compacted output is still static-shape under JAX. Use
          `num_vertices` and `num_faces` to know how many rows are valid.
        - Gradients are only meaningful with respect to the current topology,
          exactly like the official DiffMC implementation.
        - `deformation` should typically be constrained to `[-0.5, 0.5]`.
    """
    dense_result = extract_dense(
        field=field,
        deformation=deformation,
        isovalue=isovalue,
    )
    compact_result = compact_dense_result(dense_result)

    vertices = compact_result.vertices - 1.0
    if normalize:
        vertices = normalize_vertices(vertices, field.shape)

    return CompactDiffMCResult(
        vertices=vertices,
        faces=compact_result.faces,
        num_vertices=compact_result.num_vertices,
        num_faces=compact_result.num_faces,
    )

extract_dense_jit = jax.jit(extract_dense)

extract_mesh_jit = jax.jit(extract_mesh, static_argnames=("normalize",))

def extract_mesh_batched(
    field_batch: Array,
    deformation_batch: Array | None = None,
    isovalue: float = 0.0,
    normalize: bool = True,
) -> CompactDiffMCResult:
    """Batch wrapper around :func:`extract_mesh`.

    Each item is extracted independently with `jax.vmap`, which keeps the code
    small and works well for classroom-sized batches.
    """
    if deformation_batch is None:
        return jax.vmap(
            lambda field: extract_mesh(field, deformation=None, isovalue=isovalue, normalize=normalize)
        )(field_batch)

    return jax.vmap(
        lambda field, deformation: extract_mesh(
            field,
            deformation=deformation,
            isovalue=isovalue,
            normalize=normalize,
        )
    )(field_batch, deformation_batch)

extract_mesh_batched_jit = jax.jit(extract_mesh_batched, static_argnames=("normalize",))


#Masking utilities
def vertex_mask(result: CompactDiffMCResult) -> Array:
    """Boolean mask selecting the valid rows of `result.vertices`."""
    return jnp.arange(result.vertices.shape[0], dtype=jnp.int32) < result.num_vertices

def face_mask(result: CompactDiffMCResult) -> Array:
    """Boolean mask selecting the valid rows of `result.faces`."""
    return jnp.arange(result.faces.shape[0], dtype=jnp.int32) < result.num_faces

def masked_vertex_sum(result: CompactDiffMCResult, values: Array) -> Array:
    """Reduce per-vertex values while ignoring the padded tail."""
    mask = vertex_mask(result)
    return jnp.sum(jnp.where(mask, values, 0.0))

def trim_to_numpy(result: CompactDiffMCResult) -> tuple[np.ndarray, np.ndarray]:
    """Convert a padded JAX result to ordinary compact NumPy arrays.

    This helper is intentionally *not* jitted. It is meant for export,
    visualization, and unit tests.
    """
    num_vertices = int(np.asarray(result.num_vertices))
    num_faces = int(np.asarray(result.num_faces))
    vertices = np.asarray(result.vertices[:num_vertices])
    faces = np.asarray(result.faces[:num_faces], dtype=np.int32)
    return vertices, faces


#IO Utilities
def save_obj(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a triangle mesh to Wavefront OBJ."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            #OBJ is 1-indexed.  Gross.
            handle.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")

def save_ascii_stl(path: str | Path, vertices: np.ndarray, faces: np.ndarray, solid_name: str = "diffmc") -> None:
    """Write a triangle mesh to ASCII STL."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"solid {solid_name}\n")
        for face in faces:
            v0, v1, v2 = vertices[face]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 0.0:
                normal = normal / norm
            else:
                normal = np.zeros(3, dtype=np.float64)

            handle.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
            handle.write("    outer loop\n")
            handle.write(f"      vertex {v0[0]} {v0[1]} {v0[2]}\n")
            handle.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
            handle.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
            handle.write("    endloop\n")
            handle.write("  endfacet\n")
        handle.write(f"endsolid {solid_name}\n")

def save_compact_result_as_obj(path: str | Path, result: CompactDiffMCResult) -> None:
    """Trim a padded result and export it as OBJ."""
    vertices, faces = trim_to_numpy(result)
    save_obj(path=path, vertices=vertices, faces=faces)

def save_compact_result_as_stl(path: str | Path, result: CompactDiffMCResult) -> None:
    """Trim a padded result and export it as STL."""
    vertices, faces = trim_to_numpy(result)
    save_ascii_stl(path=path, vertices=vertices, faces=faces)


#Autodiff utils
def masked_vertex_l2_loss(
    field: Array,
    deformation: Array | None = None,
    isovalue: float = 0.0,
    normalize: bool = True,
) -> Array:
    """A simple differentiable scalar loss over the active vertices."""
    result = extract_mesh_jit(
        field=field,
        deformation=deformation,
        isovalue=isovalue,
        normalize=normalize,
    )
    mask = vertex_mask(result)[:, None]
    return jnp.sum(jnp.where(mask, result.vertices**2, 0.0))

def reverse_mode_field_gradient(
    field: Array,
    deformation: Array | None = None,
    isovalue: float = 0.0,
    normalize: bool = True,
) -> Array:
    """Use reverse mode for the common 'scalar loss w.r.t. many grid values' case."""
    return jax.grad(masked_vertex_l2_loss, argnums=0)(
        field,
        deformation,
        isovalue,
        normalize,
    )

def reverse_mode_deformation_gradient(
    field: Array,
    deformation: Array,
    isovalue: float = 0.0,
    normalize: bool = True,
) -> Array:
    """Use reverse mode for the common 'scalar loss w.r.t. many deformation variables' case."""
    return jax.grad(masked_vertex_l2_loss, argnums=1)(
        field,
        deformation,
        isovalue,
        normalize,
    )

def forward_mode_mesh_jvp(
    field: Array,
    field_tangent: Array,
    deformation: Array | None = None,
    deformation_tangent: Array | None = None,
    isovalue: float = 0.0,
    normalize: bool = True,
) -> tuple[CompactDiffMCResult, CompactDiffMCResult]:
    """Directional derivative of the compact padded mesh.

    Forward mode is a good fit when you want a Jacobian-vector product, for
    example to probe a single optimization direction or to build finite-
    dimensional linearizations in class.
    """
    if deformation is None:
        return jax.jvp(
            lambda field_value: extract_mesh_jit(
                field=field_value,
                deformation=None,
                isovalue=isovalue,
                normalize=normalize,
            ),
            (field,),
            (field_tangent,),
        )

    if deformation_tangent is None:
        deformation_tangent = jnp.zeros_like(deformation)

    return jax.jvp(
        lambda field_value, deformation_value: extract_mesh_jit(
            field=field_value,
            deformation=deformation_value,
            isovalue=isovalue,
            normalize=normalize,
        ),
        (field, deformation),
        (field_tangent, deformation_tangent),
    )


#examples
def sphere_sdf(points: jax.Array, radius: float = 0.5) -> jax.Array:
    return jnp.linalg.norm(points, axis=-1) - radius

def torus_sdf(points: jax.Array, major_radius: float = 0.55, minor_radius: float = 0.18) -> jax.Array:
    q = jnp.stack(
        [
            jnp.linalg.norm(points[..., :2], axis=-1) - major_radius,
            points[..., 2],
        ],
        axis=-1,
    )
    return jnp.linalg.norm(q, axis=-1) - minor_radius

def rounded_box_sdf(points: jax.Array, half_extent: float = 0.45, radius: float = 0.08) -> jax.Array:
    q = jnp.abs(points) - half_extent
    outside = jnp.maximum(q, 0.0)
    inside = jnp.minimum(jnp.max(q, axis=-1), 0.0)
    return jnp.linalg.norm(outside, axis=-1) + inside - radius

def demo_sphere_extraction(
    shape: tuple[int, int, int] = (32, 32, 32),
) -> tuple[jax.Array, jax.Array]:
    grid = make_regular_grid(shape=shape, bounds_min=(-1.0, -1.0, -1.0), bounds_max=(1.0, 1.0, 1.0))
    field = sphere_sdf(grid)
    result = extract_mesh_jit(field=field, deformation=None, isovalue=0.0, normalize=True)
    return trim_to_numpy(result)

def demo_reverse_mode_gradient(shape: tuple[int, int, int] = (16, 16, 16)) -> jax.Array:
    grid = make_regular_grid(shape=shape, bounds_min=(-1.0, -1.0, -1.0), bounds_max=(1.0, 1.0, 1.0))
    field = sphere_sdf(grid)
    return reverse_mode_field_gradient(field=field)

def demo_forward_mode_directional_derivative(shape: tuple[int, int, int] = (16, 16, 16)):
    grid = make_regular_grid(shape=shape, bounds_min=(-1.0, -1.0, -1.0), bounds_max=(1.0, 1.0, 1.0))
    field = sphere_sdf(grid)
    tangent = jnp.zeros_like(field).at[shape[0] // 2, shape[1] // 2, shape[2] // 2].set(1.0)
    return forward_mode_mesh_jvp(field=field, field_tangent=tangent)


#Tests
def test_jit_matches_eager_on_sphere() -> None:
    grid = make_regular_grid((16, 16, 16), bounds_min=(-1.0, -1.0, -1.0), bounds_max=(1.0, 1.0, 1.0))
    field = sphere_sdf(grid)
    eager = extract_mesh(field=field, deformation=None, isovalue=0.0, normalize=True)
    jitted = extract_mesh_jit(field=field, deformation=None, isovalue=0.0, normalize=True)

    assert int(np.asarray(eager.num_vertices)) == int(np.asarray(jitted.num_vertices))
    assert int(np.asarray(eager.num_faces)) == int(np.asarray(jitted.num_faces))
    np.testing.assert_allclose(np.asarray(eager.vertices), np.asarray(jitted.vertices), atol=1e-6)
    np.testing.assert_array_equal(np.asarray(eager.faces), np.asarray(jitted.faces))

def test_reverse_mode_gradient_matches_finite_difference() -> None:
    grid = make_regular_grid((8, 8, 8), bounds_min=(-1.0, -1.0, -1.0), bounds_max=(1.0, 1.0, 1.0))
    field = sphere_sdf(grid)

    gradient = jax.grad(masked_vertex_l2_loss, argnums=0)(field, None, 0.0, True)
    index = (2, 3, 3)
    epsilon = 1e-4
    finite_difference = (
        masked_vertex_l2_loss(field.at[index].add(epsilon), None, 0.0, True)
        - masked_vertex_l2_loss(field.at[index].add(-epsilon), None, 0.0, True)
    ) / (2.0 * epsilon)

    assert np.isfinite(np.asarray(gradient[index]))
    np.testing.assert_allclose(
        np.asarray(gradient[index]),
        np.asarray(finite_difference),
        rtol=3e-2,
        atol=3e-2,
    )

def test_deformation_gradient_matches_finite_difference() -> None:
    grid = make_regular_grid((8, 8, 8), bounds_min=(-1.0, -1.0, -1.0), bounds_max=(1.0, 1.0, 1.0))
    field = sphere_sdf(grid)
    deformation = jnp.zeros(field.shape + (3,), dtype=field.dtype)

    def deformation_loss(deformation_value: jax.Array) -> jax.Array:
        return masked_vertex_l2_loss(field, deformation_value, 0.0, True)

    gradient = jax.grad(deformation_loss)(deformation)
    index = (1, 3, 4, 2)
    epsilon = 1e-3
    finite_difference = (
        deformation_loss(deformation.at[index].add(epsilon))
        - deformation_loss(deformation.at[index].add(-epsilon))
    ) / (2.0 * epsilon)

    assert np.isfinite(np.asarray(gradient[index]))
    np.testing.assert_allclose(
        np.asarray(gradient[index]),
        np.asarray(finite_difference),
        rtol=3e-2,
        atol=3e-2,
    )

def test_extracted_sphere_is_watertight() -> None:
    grid = make_regular_grid((16, 16, 16), bounds_min=(-1.0, -1.0, -1.0), bounds_max=(1.0, 1.0, 1.0))
    field = sphere_sdf(grid)
    result = extract_mesh_jit(field=field, deformation=None, isovalue=0.0, normalize=True)
    _, faces = trim_to_numpy(result)

    undirected_edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0),
        axis=1,
    )
    edge_counter = Counter(map(tuple, undirected_edges))
    assert edge_counter
    assert all(count == 2 for count in edge_counter.values())

def run_all_tests() -> None:
    test_jit_matches_eager_on_sphere()
    test_reverse_mode_gradient_matches_finite_difference()
    test_deformation_gradient_matches_finite_difference()
    test_extracted_sphere_is_watertight()
    print("All DiffMC JAX tests passed.")









def build_sphere_example(
    shape: tuple[int, int, int] = (32, 32, 32),
    radius: float = 0.5,
) -> tuple[Array, Array, CompactDiffMCResult, np.ndarray, np.ndarray]:
    """Build a sphere field, extract a mesh, and return both padded and trimmed forms."""
    grid = make_regular_grid(
        shape=shape,
        bounds_min=(-1.0, -1.0, -1.0),
        bounds_max=(1.0, 1.0, 1.0),
    )
    field = sphere_sdf(grid, radius=radius)
    result = extract_mesh_jit(
        field=field,
        deformation=None,
        isovalue=0.0,
        normalize=True,
    )
    vertices, faces = trim_to_numpy(result)

    #debug code
    #print("vertices:", vertices.shape)
    #print("faces:", faces.shape)
    #print("First few vertices:")
    #print(vertices[:5])
    #print("First few faces:")
    #print(faces[:5])

    return grid, field, result, vertices, faces


def export_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_dir: str | Path = "diffmc_demo_out",
    stem: str = "sphere",
) -> Path:
    """Export both OBJ and STL for a trimmed mesh."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    obj_path = output_path / f"{stem}.obj"
    stl_path = output_path / f"{stem}.stl"

    save_obj(obj_path, vertices, faces)
    save_ascii_stl(stl_path, vertices, faces)

    print("Wrote", obj_path)
    print("Wrote", stl_path)
    return output_path


def run_reverse_mode_example(field: Array) -> Array:
    """Differentiate a simple scalar loss with respect to the input field."""
    field_grad = reverse_mode_field_gradient(field=field)
    print(
        "reverse-mode field gradient:",
        field_grad.shape,
        float(jnp.min(field_grad)),
        float(jnp.max(field_grad)),
    )
    return field_grad


def run_forward_mode_example(field: Array) -> tuple[CompactDiffMCResult, CompactDiffMCResult]:
    """Take a single JVP through the extractor."""
    center_index = tuple(dim // 2 for dim in field.shape)
    field_tangent = jnp.zeros_like(field).at[center_index].set(1.0)
    primal_mesh, tangent_mesh = forward_mode_mesh_jvp(
        field=field,
        field_tangent=field_tangent,
    )
    print("primal num vertices:", int(np.asarray(primal_mesh.num_vertices)))
    print("primal num faces:", int(np.asarray(primal_mesh.num_faces)))
    print("tangent vertices shape:", tangent_mesh.vertices.shape)
    print("tangent faces shape:", tangent_mesh.faces.shape)
    return primal_mesh, tangent_mesh


def run_demo(
    output_dir: str | Path = "diffmc_demo_out",
    shape: tuple[int, int, int] = (32, 32, 32),
    radius: float = 0.5,
    run_tests: bool = False,
) -> None:
    """Run a small end-to-end DiffMC demo."""
    _, field, _, vertices, faces = build_sphere_example(shape=shape, radius=radius)
    export_mesh(vertices=vertices, faces=faces, output_dir=output_dir, stem="sphere")
    run_reverse_mode_example(field=field)
    run_forward_mode_example(field=field)
    if run_tests:
        run_all_tests()


def _parse_shape(shape_values: list[int]) -> tuple[int, int, int]:
    if len(shape_values) != 3:
        raise ValueError(f"--shape expects exactly 3 integers, got {shape_values}")
    return int(shape_values[0]), int(shape_values[1]), int(shape_values[2])

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import trimesh
from scipy import ndimage


def load_obj_to_levelset_fast(
    obj_path: str | Path,
    max_dim: int = 128,
) -> tuple[jnp.ndarray, np.ndarray]:
    """
    Convert an OBJ into a voxel-grid level set much faster than
    trimesh.proximity.signed_distance.

    Returns
    -------
    field:
        Signed grid with negative values inside and positive values outside.
    voxel_transform:
        4x4 transform that maps voxel coordinates back to world coordinates.
    """
    mesh = trimesh.load_mesh(obj_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    #Basic cleanup
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    pitch = float(mesh.extents.max() / (max_dim - 1))

    #Fast voxelization path
    vox = mesh.voxelized(pitch).fill()
    occ = vox.matrix.astype(bool)

    #Negative inside, positive outside
    outside = ndimage.distance_transform_edt(~occ)
    inside = ndimage.distance_transform_edt(occ)
    field = jnp.asarray(outside - inside, dtype=jnp.float32)

    return field, np.asarray(vox.transform, dtype=np.float32)


def unnormalize_vertices(
    vertices: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> np.ndarray:
    """
    Map normalized vertices in [0, 1]^3 back to world coordinates.
    """
    return bounds_min[None, :] + vertices * (bounds_max - bounds_min)[None, :]

def main(obj_input) -> None:
    if not obj_input:
        parser = argparse.ArgumentParser(description="Single-file JAX DiffMC demo and test runner.")
        parser.add_argument(
            "--output-dir",
            type=str,
            default="diffmc_demo_out",
            help="Directory used for OBJ/STL output.",
        )
        parser.add_argument(
            "--shape",
            nargs=3,
            type=int,
            default=(32, 32, 32),
            metavar=("NX", "NY", "NZ"),
            help="Grid shape for the demo extraction.",
        )
        parser.add_argument(
            "--radius",
            type=float,
            default=0.5,
            help="Sphere radius used by the demo SDF.",
        )
        parser.add_argument(
            "--run-tests",
            action="store_true",
            help="Run the lightweight built-in tests after the demo.",
        )
        parser.add_argument(
            "--tests-only",
            action="store_true",
            help="Skip the demo and run only the built-in tests.",
        )
        args, _ = parser.parse_known_args()

        shape = _parse_shape(list(args.shape))
        if args.tests_only:
            run_all_tests()
            return

        run_demo(
            output_dir=args.output_dir,
            shape=shape,
            radius=float(args.radius),
            run_tests=bool(args.run_tests),
        )
    else:

        field, voxel_transform = load_obj_to_levelset_fast(
            obj_input,
            max_dim=128,
        )

        result = extract_mesh_jit(
            field=field,
            deformation=None,
            isovalue=0.0,
            normalize=False,
        )

        vertices, faces = trim_to_numpy(result)
        vertices_world = trimesh.transform_points(vertices, voxel_transform)

        save_obj("out.obj", vertices_world, faces)
        save_ascii_stl("out.stl", vertices_world, faces)


if __name__ == "__main__":
    main(obj_input=None)