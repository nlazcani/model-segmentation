import bpy
import bgl
import gpu
import blf
import bmesh
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from multiprocessing import Pool
from line_profiler import LineProfiler
import time

start = time.time()

D = bpy.data
C = bpy.context
bpy.ops.import_mesh.stl(
    filepath="D:\\Proyectos\\blender\\mesh_segmentation\\Armadillo.stl"
)
obj = D.objects["Armadillo"]
C.view_layer.objects.active = obj
bpy.ops.view3d.snap_cursor_to_active()
bpy.ops.view3d.view_all(center=True)

# get center point
x, y, z = [sum([v.co[i] for v in obj.data.vertices]) for i in range(3)]
count = float(len(obj.data.vertices))
center = obj.matrix_world @ (Vector((x, y, z)) / count)
print(center)


def draw_line_3d(color, start, end):
    shader = gpu.shader.from_builtin("3D_UNIFORM_COLOR")
    batch = batch_for_shader(shader, "LINES", {"pos": [start, end]})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def draw_point_3d(color, pos):
    shader = gpu.shader.from_builtin("3D_UNIFORM_COLOR")
    scale = (view_location() - pos).length / 400.0
    base_coord = pos
    h_coords = [base_coord + (Vector(c) * scale) for c in create_highlight()]
    batch = batch_for_shader(shader, "TRIS", {"pos": h_coords})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def create_highlight():
    # it is based on UV sphere triangulated for the GPU rendering
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    return [v.co.to_tuple() for f in bm.faces for v in f.verts]


def get_sphere_points(distance):
    # it is based on UV sphere triangulated for the GPU rendering
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, diameter=distance)
    return [v.co.to_tuple() for v in bm.verts]


# View location in 3D
def view_location():
    camera_info = C.space_data.region_3d.view_matrix.inverted()
    return camera_info.translation


def draw_callback_3d(operator, context, points, center):

    # 80% alpha, 2 pixel width line
    bgl.glEnable(bgl.GL_BLEND)
    bgl.glEnable(bgl.GL_LINE_SMOOTH)
    bgl.glEnable(bgl.GL_DEPTH_TEST)

    # green line
    draw_line_3d((0.0, 1.0, 0.0, 0.7), points[0], center)

    # red point
    for p in points:
        draw_point_3d((1.0, 0.0, 0.0, 0.5), p)

    # restore opengl defaults
    bgl.glLineWidth(1)
    bgl.glDisable(bgl.GL_BLEND)
    bgl.glDisable(bgl.GL_LINE_SMOOTH)
    bgl.glEnable(bgl.GL_DEPTH_TEST)


points = [center + Vector(c) for c in get_sphere_points(250)]


group = obj.vertex_groups.new(name="1")
group = obj.vertex_groups.new(name="2")
group = obj.vertex_groups.new(name="3")
# Assign material to vertex group
red = bpy.data.materials.new(name="red")
obj.data.materials.append(red)
yellow = bpy.data.materials.new(name="yellow")
obj.data.materials.append(yellow)
blue = bpy.data.materials.new(name="blue")
obj.data.materials.append(blue)

bpy.data.materials["red"].diffuse_color = (0.938686, 0.0069953, 0.0122865, 1)
bpy.data.materials["yellow"].diffuse_color = (1, 0.346704, 0.0116123, 1)
bpy.data.materials["blue"].diffuse_color = (0.0069953, 0.0122865, 0.938686, 1)


def is_visible(p, point_candi):
    dp = p.normal.dot(center - point_candi)
    return dp < 0


def is_really_visible(pr, point_candi):
    p = mesh.polygons[pr[0]]
    visible = False
    if pr[1]:
        _, _, _, index = obj.ray_cast(point_candi, p.center - point_candi)
        visible = p.index == index
    return (p.index, visible)


mesh = obj.data
stats = []
import itertools

# TODO: not used yet
# detecting groups of polygons (dot product)
vert_pol = [(v, p) for p in mesh.polygons for v in p.vertices]
vert_pol.sort(key=lambda x: x[0])
vert_pol_grouped = {key:[num for _, num in value]
    for key, value in itertools.groupby(vert_pol, lambda x: x[0])}
poly_neighbors = [ (poly.index, [p for p in vert_pol_grouped[v] if p != poly]) for poly in mesh.polygons for v in poly.vertices ]
poly_neighbors.sort(key=lambda x: x[0])
poly_neighbors_grouped = {key:{el for _, num in value for el in num}
    for key, value in itertools.groupby(poly_neighbors, lambda x: x[0])}

visited_polygons =  [False] * len(mesh.polygons)
groups = dict()
def visit_neighbor(polygon, group):
    if not visited_polygons[polygon.index] and mesh.polygons[group].normal.dot(polygon.normal) > 0.97:
        visited_polygons[polygon.index] = True
        if group in groups:
            groups[group].append(polygon)
        else:
            groups[group] = [polygon]
        for child in poly_neighbors_grouped[polygon.index]:
            visit_neighbor(child, group)

[visit_neighbor(pn, p.index) for p in mesh.polygons for pn in poly_neighbors_grouped[p.index] if not visited_polygons[p.index]]

print(f"elementos por grupo: {len(mesh.polygons)/len(groups)}")

def check_visibles(point_candi, sub):
    polys = [p for p in mesh.polygons if p.index % sub == 0]
    polys_visibles = [[p.index, is_visible(p, point_candi)] for p in polys]
    polys_real_visibles = [is_really_visible(x, point_candi) for x in polys_visibles]
    visible = {
        e for pr in polys_real_visibles for e in mesh.polygons[pr[0]].vertices if pr[1]
    }
    return visible


def check_candidate_points(points, mesh, stats, check_visibles):
    for id, point_candi in enumerate(points):
        sub = 100
        visible = check_visibles(point_candi, sub)
        stats.append((id, visible))
        print(f"{id}, {len(visible)*100/(len(mesh.vertices)/sub)}")

    stats.sort(key=lambda tup: len(tup[1]) / len(mesh.vertices), reverse=True)
    visible = check_visibles(points[stats[0][0]], 1)
    v_oposite = 2 * center - points[39]
    oposite = check_visibles(v_oposite, 1)
    not_visible = [
        v.index
        for v in mesh.vertices
        if v.index not in visible and v.index not in oposite
    ]
    return visible, not_visible, oposite, stats[0][0]


visible, not_visible, oposite, id_condidate = check_candidate_points(
    points, mesh, stats, check_visibles
)


# lp = LineProfiler()
# lp_wrapper = lp(check_candidate_points)
# visible, not_visible = lp_wrapper(points, mesh, stats, check_visibles)
# lp.print_stats()

vertex_group = obj.vertex_groups[0]
vertex_group.add(list(visible), 1.0, "ADD")
vertex_group = obj.vertex_groups[1]
vertex_group.add(list(oposite), 1.0, "ADD")
vertex_group = obj.vertex_groups[2]
vertex_group.add(not_visible, 1.0, "ADD")
print(id_condidate)
args = (None, C, [points[id_condidate]], center)
_handle = bpy.types.SpaceView3D.draw_handler_add(
    draw_callback_3d, args, "WINDOW", "POST_VIEW"
)

# bpy.ops.object.mode_set(mode='EDIT')
# bpy.ops.object.vertex_group_deselect()

# bpy.ops.object.vertex_group_set_active(group=str(0))
# bpy.ops.object.vertex_group_select()
# bpy.context.object.active_material_index = 0
# bpy.ops.object.material_slot_assign()
# bpy.ops.object.vertex_group_deselect()

end = time.time()
print(end - start)

# | sub | time |
# | 1   | 518  |
# | 10  | 70   |
# | 100 | 21   |
