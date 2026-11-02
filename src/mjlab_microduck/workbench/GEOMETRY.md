# Private compiled-scene projection foundation

`geometry.py` is deliberately unused by existing producers. It is not a wire
schema or an advertised capability. Current observer, evaluation, legacy viewer,
public schemas and catalog bytes remain unchanged. Integration requires a new
versioned producer/consumer contract and separate end-to-end qualification.

## Interface

`build_projection(mujoco_model, robot_root_body_id=<trusted non-world body ID>)`
returns a frozen `GeometryProjection`. The caller owns model identity and must
rebuild after changing model geometry, fixed transforms or materials. The builder
does not allocate `MjData`, step, reset or mutate a simulation/model.

- `geometry_json`: immutable canonical JSON text; `export()` returns a detached
  mutable copy containing geometry, shared meshes, materials, focus slot IDs,
  source geom count, explicit `dense-display-slot` identity and mapping digest.
- `source_geom_count`, `mapping_digest`: source cardinality and SHA-256 identity
  covering exported geometry, trusted root, source membership and constant poses.
- `dynamic_mapping`: immutable `(display_slot, actual_source_geom_id)` tuples.
- `source_membership`: immutable tuples of original source geom IDs per display
  slot, including all members of each baked static terrain chunk.
- Private immutable constant positions/matrices accompany this mapping. Mutating
  an exported JSON object does not change subsequent exports or captures.
- `capture(positions, matrices, environment_indices=[...])`: accepts actual NumPy
  or Torch floating arrays of shapes `[E, source_count, 3]` and
  `[E, source_count, 3, 3]`. The explicit ordered list must contain 1–9 distinct
  valid integer environment indices. It returns a fresh frame with mapping digest,
  environment indices, slot count and packed position/matrix buffers. It never
  selects synthetic environment states or advances the environment.

Both NumPy and Torch gather environment AND dynamic source indices before any
CPU transfer. No full source tensor is cloned. Fixed poses remain on CPU; Torch
source-index tensors use a lock-protected two-device LRU, created only on a device
already supplied by the caller. Capture does not initialize CUDA by itself.
Caller synchronization is required if a simulator writes source arrays
concurrently: this helper does not promise an atomic simulation snapshot.

## Geometry and encoding

Only existing hidden visual group 3 is skipped. Dynamic classification follows
all body ancestors with DOFs or mocap control, including fixed child bodies.
Robot focus follows the explicitly trusted robot body's ancestry, excluding
terrain and independent objects. Dynamic objects and robot geometry retain their
actual primitives and shared compiled mesh assets. Static non-terrain primitives
and planes retain their compiled sizes and constant world poses.

Non-robot fixed boxes and heightfields are transformed into world-space meshes.
Boxes use eight corners and twelve triangles. Heightfields use actual compiled
row/column/data/size: X is column, Y is row, Z is sample times elevation scale;
the cell diagonal connects `(row,col)` to `(row+1,col+1)`. Side walls and the
bottom at negative base depth are included. Geometry has no decimation, ROI
omission, invented fill or physics modification.

Terrain partition clips each source surface against the deterministic world XY
16m grid and groups resulting pieces by grid cell and material ID. Triangles
already inside a cell are retained; crossing triangles are clipped against its
four XY half-planes and triangulated without decimation. No internal cut walls
are introduced. Exact-equality vertex deduplication within each source/cell
reduces expansion without merging colors across different sources. The decoded
XY extent of every terrain mesh is at most16m (tests allow2e-5m for float32
arithmetic); Z extent remains the real source surface. Cell boundaries are
multiples of16m, exactly representable in float32 throughout the supported
coordinate range. Ray/surface coordinates otherwise round to float32 as below.

A source geom legitimately occurs in multiple chunk `source_membership` entries;
IDs identify original sources, not distinct invented source geometry. Each chunk
lists each contributing source once. The union equals all nonhidden source IDs;
shared boundaries contain no duplicated positive-area surface. The mapping
digest includes this membership and the actual clipped geometry.

Before surface expansion a conservative transformed source bounding-volume
cell count above256 refuses with `SPATIAL_CELL_LIMIT`. This bounds enumeration,
even for pathological huge/rotated fields; a very sparse diagonal source may
therefore be refused conservatively. Before clipped polygon allocation the
builder checks planned slot occupancy and conservative per-cell/aggregate
vertex, triangle and encoded-byte bounds. A triangle/rectangle intersection
has at most7vertices/5triangles. Oversized outputs refuse without truncation.

All vertices and poses use packed little-endian float32 buffers, base64 encoded.
Matrices are row-major 3×3. Compiled mesh vertices retain their float32 bits;
transformed fixed terrain and float64 input poses round to float32. Triangle
indices are little-endian unsigned 16-bit when vertex count is at most 65536,
otherwise unsigned 32-bit. Chunk meshes additionally contain `colorsBase64`,
one little-endian float32 RGBA tuple per vertex, matching vertex order. Colors
are the compiled geom override, or material RGBA when the geom uses default RGBA.
The chunk geometry's white RGBA is an identity multiplier, not its real color.

Material IDs and available emission/specular/shininess/reflectance/metallic/
roughness scalars are retained; chunk grouping keeps material boundaries.
`textureOmitted` records material texture use. Texture data, UVs, mesh normals,
texture URLs/paths and executable/plugin configuration are not exported. A future
renderer must support vertex colors, material properties, shared mesh references,
packed pose buffers, display-slot identity and world-space terrain. Texture and
normal-dependent shading parity is **not qualified**. Infinite-plane rendering
semantics remain a renderer concern; no extra finite terrain is synthesized.
No existing renderer is claimed to understand this payload.

Validated visible numeric fields are checked in bounded batches after the source
cap. Box world corners and cell ranges are computed once; boxes wholly contained
in a cell/material are reserved together before allocating their vertex/index/
color buffers. Crossing boxes and heightfields retain the exact clipper. Batch
storage changes buffer ordering deterministically; source memberships, display
slot ordering and mapped shapes remain explicit, and the digest covers the
resulting buffers. No validation or expansion bound is bypassed by batching.

## Bounds and refusals

Source count ≤32768; bodies ≤65536; display slots ≤256; aggregate unique exported
mesh vertices ≤1000000; triangles ≤2000000; each mesh's vertex and index component
counts ≤1500000; encoded geometry ≤64 MiB; frames ≤262144 bytes. Source, component,
aggregate and predicted encoded allocation limits are checked before expanding
terrain arrays. Geometry is never truncated or flattened to pass a cap.

`ProjectionError` carries a specific private code, including
`SOURCE_GEOMETRY_LIMIT`, `SPATIAL_CELL_LIMIT`, `CLIP_TOPOLOGY`,
`DISPLAY_SLOT_LIMIT`, `MESH_BUDGET`,
`MESH_COMPONENT_LIMIT`, `GEOMETRY_BYTES`, `FRAME_BUDGET`, `ROBOT_ROOT`,
`ROBOT_GEOMETRY_EMPTY`, `BODY_ANCESTRY`, `UNSUPPORTED_GEOMETRY`,
`DYNAMIC_HEIGHTFIELD_UNSUPPORTED`, `ROBOT_HEIGHTFIELD_UNSUPPORTED`,
`MESH_REFERENCE`, `MESH_SLICE`, `MESH_INDEX`, `HFIELD_REFERENCE`,
`HFIELD_SHAPE`, `HFIELD_DATA`, `MATERIAL_REFERENCE`, `NONFINITE_GEOMETRY`,
`POSE_TYPE`, `POSE_SHAPE`, `ENVIRONMENT_INDICES` and `NONFINITE_POSE`.
Coordinates and pose components must also have magnitude ≤1000000.

The current registered training Rough configuration is observed as 5×5, even at
4096 origins: an existing factory shares its terrain configuration with play
registration, which mutates its dimensions to 5×5. The original generator mode is already
random (`curriculum=False`); this work does not infer or restore a true mode. This module does not fix or depend on that
mutation as intended behavior. The owning tests distinguish unchanged-factory
metadata evidence from explicitly copied 10×20 stress input. Neither constitutes
training/evaluation quality, all-recipe functionality, 20 Hz overhead, GPU,
interactive rendering or public launch qualification.
