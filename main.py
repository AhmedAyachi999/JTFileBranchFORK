from pxr import Usd, UsdGeom, UsdLux, Gf

output_usd = r"C:\Users\Ahmed\Downloads\two_worker_sims.usda"

# --------------------------------------------------
# CREATE STAGE
# --------------------------------------------------
stage = Usd.Stage.CreateNew(output_usd)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
stage.SetStartTimeCode(1)
stage.SetEndTimeCode(120)
stage.SetTimeCodesPerSecond(24)

world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def set_color(gprim, color):
    gprim.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])


def make_xform(path, translate=(0, 0, 0)):
    xf = UsdGeom.Xform.Define(stage, path)
    xformable = UsdGeom.Xformable(xf.GetPrim())
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translate))
    return xf


def make_cube(path, size=1.0, translate=(0, 0, 0), scale=(1, 1, 1), color=(0.8, 0.8, 0.8)):
    prim = UsdGeom.Cube.Define(stage, path)
    prim.CreateSizeAttr(size)
    xf = UsdGeom.Xformable(prim.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xf.AddScaleOp().Set(Gf.Vec3f(*scale))
    set_color(prim, color)
    return prim


def make_sphere(path, radius=1.0, translate=(0, 0, 0), color=(0.8, 0.8, 0.8)):
    prim = UsdGeom.Sphere.Define(stage, path)
    prim.CreateRadiusAttr(radius)
    xf = UsdGeom.Xformable(prim.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    set_color(prim, color)
    return prim


def make_cylinder(path, radius=0.2, height=1.0, translate=(0, 0, 0), rotate=(0, 0, 0), color=(0.8, 0.8, 0.8)):
    prim = UsdGeom.Cylinder.Define(stage, path)
    prim.CreateRadiusAttr(radius)
    prim.CreateHeightAttr(height)
    xf = UsdGeom.Xformable(prim.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*rotate))
    set_color(prim, color)
    return prim


def build_connected_room(root_path,
                         width=24.0,
                         depth=16.0,
                         height=8.0,
                         wall_thickness=0.12,
                         floor_thickness=0.10,
                         ceiling_thickness=0.10):
    half_w = width / 2.0
    half_d = depth / 2.0
    half_h = height / 2.0

    wall_color = (0.93, 0.93, 0.91)
    floor_color = (0.72, 0.68, 0.60)
    ceiling_color = (0.96, 0.96, 0.95)

    make_cube(
        f"{root_path}/Floor",
        size=1.0,
        translate=(0, -floor_thickness / 2.0, 0),
        scale=(half_w, floor_thickness / 2.0, half_d),
        color=floor_color,
    )

    make_cube(
        f"{root_path}/Ceiling",
        size=1.0,
        translate=(0, height + ceiling_thickness / 2.0, 0),
        scale=(half_w, ceiling_thickness / 2.0, half_d),
        color=ceiling_color,
    )

    make_cube(
        f"{root_path}/BackWall",
        size=1.0,
        translate=(-half_w - wall_thickness / 2.0, half_h, 0),
        scale=(wall_thickness / 2.0, half_h, half_d + wall_thickness),
        color=wall_color,
    )

    make_cube(
        f"{root_path}/LeftWall",
        size=1.0,
        translate=(0, half_h, -half_d - wall_thickness / 2.0),
        scale=(half_w + wall_thickness, half_h, wall_thickness / 2.0),
        color=wall_color,
    )

    make_cube(
        f"{root_path}/RightWall",
        size=1.0,
        translate=(0, half_h, half_d + wall_thickness / 2.0),
        scale=(half_w + wall_thickness, half_h, wall_thickness / 2.0),
        color=wall_color,
    )


def build_simulation(root_path):
    # room shell
    build_connected_room(root_path)

    # local room light
    ceiling_light = UsdLux.SphereLight.Define(stage, f"{root_path}/CeilingLight")
    ceiling_light.CreateIntensityAttr(3500)
    ceiling_light.CreateRadiusAttr(0.45)
    ceiling_light_xf = UsdGeom.Xformable(ceiling_light.GetPrim())
    ceiling_light_xf.AddTranslateOp().Set(Gf.Vec3d(0, 7.2, 0))

    # shelf
    make_xform(f"{root_path}/Shelf")

    make_cube(f"{root_path}/Shelf/LeftSide",  size=1.0, translate=(-8.8, 3.1, -1.8), scale=(0.14, 6.2, 0.14), color=(0.35, 0.22, 0.12))
    make_cube(f"{root_path}/Shelf/RightSide", size=1.0, translate=(-8.8, 3.1,  1.8), scale=(0.14, 6.2, 0.14), color=(0.35, 0.22, 0.12))
    make_cube(f"{root_path}/Shelf/Back",      size=1.0, translate=(-9.4, 3.1,  0.0), scale=(0.06, 6.2, 3.8), color=(0.88, 0.87, 0.84))

    make_cube(f"{root_path}/Shelf/Board1", size=1.0, translate=(-8.0, 1.0, 0), scale=(1.8, 0.12, 3.5), color=(0.42, 0.28, 0.16))
    make_cube(f"{root_path}/Shelf/Board2", size=1.0, translate=(-8.0, 3.0, 0), scale=(1.8, 0.12, 3.5), color=(0.42, 0.28, 0.16))
    make_cube(f"{root_path}/Shelf/Board3", size=1.0, translate=(-8.0, 5.0, 0), scale=(1.8, 0.12, 3.5), color=(0.42, 0.28, 0.16))

    make_cube(f"{root_path}/Shelf/ObjectA", size=1.0, translate=(-7.5, 1.4, -1.0), scale=(0.35, 0.35, 0.25), color=(0.75, 0.20, 0.20))
    make_cube(f"{root_path}/Shelf/ObjectB", size=1.0, translate=(-7.7, 1.35, 1.0), scale=(0.20, 0.25, 0.20), color=(0.20, 0.40, 0.75))
    make_cube(f"{root_path}/Shelf/ObjectC", size=1.0, translate=(-7.3, 5.35, 0.7), scale=(0.28, 0.18, 0.22), color=(0.80, 0.70, 0.20))

    # table
    make_cube(f"{root_path}/TableTop", size=1.0, translate=(8, 2.1, 0), scale=(4.2, 0.18, 2.4), color=(0.45, 0.28, 0.15))
    make_cube(f"{root_path}/TableLeg1", size=1.0, translate=(4.3, 1.0, -1.9), scale=(0.16, 2.1, 0.16), color=(0.28, 0.18, 0.10))
    make_cube(f"{root_path}/TableLeg2", size=1.0, translate=(11.7, 1.0, -1.9), scale=(0.16, 2.1, 0.16), color=(0.28, 0.18, 0.10))
    make_cube(f"{root_path}/TableLeg3", size=1.0, translate=(4.3, 1.0,  1.9), scale=(0.16, 2.1, 0.16), color=(0.28, 0.18, 0.10))
    make_cube(f"{root_path}/TableLeg4", size=1.0, translate=(11.7, 1.0,  1.9), scale=(0.16, 2.1, 0.16), color=(0.28, 0.18, 0.10))

    make_cube(f"{root_path}/TableCup", size=1.0, translate=(9.5, 2.45, 0.7), scale=(0.16, 0.25, 0.16), color=(0.90, 0.90, 0.92))
    make_cube(f"{root_path}/TableNotebook", size=1.0, translate=(7.2, 2.28, -0.6), scale=(0.45, 0.03, 0.32), color=(0.18, 0.24, 0.55))

    # moving item
    item = make_cube(f"{root_path}/Item", size=1.0, translate=(-7.2, 3.45, 0.0), scale=(0.45, 0.30, 0.35), color=(0.82, 0.55, 0.18))
    item_xf = UsdGeom.Xformable(item.GetPrim())
    item_t = item_xf.GetOrderedXformOps()[0]

    # worker
    worker = UsdGeom.Xform.Define(stage, f"{root_path}/Worker")
    worker_xf = UsdGeom.Xformable(worker.GetPrim())
    worker_t = worker_xf.AddTranslateOp()

    worker_t.Set(Gf.Vec3d(-3.5, 0.0, 0.0), 1)
    worker_t.Set(Gf.Vec3d(-5.5, 0.0, 0.0), 24)
    worker_t.Set(Gf.Vec3d(-6.0, 0.0, 0.0), 36)
    worker_t.Set(Gf.Vec3d(-6.0, 0.0, 0.0), 56)
    worker_t.Set(Gf.Vec3d(-5.5, 0.0, 0.0), 64)
    worker_t.Set(Gf.Vec3d(-4.5, 0.0, 0.0), 74)
    worker_t.Set(Gf.Vec3d(-3.0, 0.0, 0.0), 84)
    worker_t.Set(Gf.Vec3d(-1.0, 0.0, 0.0), 94)
    worker_t.Set(Gf.Vec3d( 1.5, 0.0, 0.0), 104)
    worker_t.Set(Gf.Vec3d( 4.0, 0.0, 0.0), 112)
    worker_t.Set(Gf.Vec3d( 6.0, 0.0, 0.0), 118)
    worker_t.Set(Gf.Vec3d( 6.0, 0.0, 0.0), 120)

    make_cube(f"{root_path}/Worker/Torso", size=1.0, translate=(0, 2.2, 0), scale=(0.9, 1.4, 0.5), color=(0.20, 0.35, 0.75))
    make_sphere(f"{root_path}/Worker/Head", radius=0.42, translate=(0, 3.7, 0), color=(0.88, 0.72, 0.58))
    make_cylinder(f"{root_path}/Worker/LeftLeg",  radius=0.16, height=1.8, translate=(-0.25, 0.9, -0.08), rotate=(0, 0, 2), color=(0.15, 0.15, 0.18))
    make_cylinder(f"{root_path}/Worker/RightLeg", radius=0.16, height=1.8, translate=( 0.25, 0.9,  0.08), rotate=(0, 0, -2), color=(0.15, 0.15, 0.18))
    make_cylinder(f"{root_path}/Worker/LeftArmStatic", radius=0.13, height=1.5, translate=(-0.75, 2.35, 0), rotate=(0, 0, 20), color=(0.88, 0.72, 0.58))

    right_arm_grp = UsdGeom.Xform.Define(stage, f"{root_path}/Worker/RightArmGroup")
    right_arm_grp_xf = UsdGeom.Xformable(right_arm_grp.GetPrim())
    right_arm_grp_t = right_arm_grp_xf.AddTranslateOp()
    right_arm_grp_r = right_arm_grp_xf.AddRotateXYZOp()

    right_arm_grp_t.Set(Gf.Vec3d(0.72, 2.95, 0.0))

    make_cylinder(
        f"{root_path}/Worker/RightArmGroup/UpperLowerArm",
        radius=0.13,
        height=1.7,
        translate=(0.65, -0.55, 0.0),
        rotate=(0, 0, -65),
        color=(0.88, 0.72, 0.58)
    )
    make_sphere(f"{root_path}/Worker/RightArmGroup/Hand", radius=0.14, translate=(1.35, -1.15, 0.0), color=(0.88, 0.72, 0.58))

    right_arm_grp_r.Set(Gf.Vec3f(0, 0, -10), 1)
    right_arm_grp_r.Set(Gf.Vec3f(0, 0, -35), 24)
    right_arm_grp_r.Set(Gf.Vec3f(0, 0, -58), 36)
    right_arm_grp_r.Set(Gf.Vec3f(0, 0, -58), 56)
    right_arm_grp_r.Set(Gf.Vec3f(0, 0, -48), 64)
    right_arm_grp_r.Set(Gf.Vec3f(0, 0, -42), 74)
    right_arm_grp_r.Set(Gf.Vec3f(0, 0, -35), 84)
    right_arm_grp_r.Set(Gf.Vec3f(0, 0, -28), 94)
    right_arm_grp_r.Set(Gf.Vec3f(0, 0, -18), 104)
    right_arm_grp_r.Set(Gf.Vec3f(0, 0,  -8), 112)
    right_arm_grp_r.Set(Gf.Vec3f(0, 0,   0), 118)
    right_arm_grp_r.Set(Gf.Vec3f(0, 0,   0), 120)

    # item animation
    item_t.Set(Gf.Vec3d(-7.2, 3.45, 0.0), 1)
    item_t.Set(Gf.Vec3d(-7.2, 3.45, 0.0), 48)

    item_t.Set(Gf.Vec3d(-7.0, 3.30, 0.0), 52)
    item_t.Set(Gf.Vec3d(-6.7, 3.15, 0.0), 56)
    item_t.Set(Gf.Vec3d(-6.2, 2.95, 0.0), 60)

    item_t.Set(Gf.Vec3d(-5.6, 2.92, 0.0), 64)
    item_t.Set(Gf.Vec3d(-4.8, 2.88, 0.0), 69)
    item_t.Set(Gf.Vec3d(-3.9, 2.84, 0.0), 74)
    item_t.Set(Gf.Vec3d(-2.8, 2.80, 0.0), 79)
    item_t.Set(Gf.Vec3d(-1.5, 2.75, 0.0), 84)
    item_t.Set(Gf.Vec3d(-0.2, 2.70, 0.0), 89)
    item_t.Set(Gf.Vec3d( 1.2, 2.66, 0.0), 94)
    item_t.Set(Gf.Vec3d( 2.7, 2.62, 0.0), 99)
    item_t.Set(Gf.Vec3d( 4.2, 2.58, 0.0), 104)
    item_t.Set(Gf.Vec3d( 5.6, 2.54, 0.0), 109)
    item_t.Set(Gf.Vec3d( 6.8, 2.50, 0.0), 114)

    item_t.Set(Gf.Vec3d( 7.6, 2.47, 0.0), 118)
    item_t.Set(Gf.Vec3d( 8.0, 2.45, 0.0), 120)


# --------------------------------------------------
# GLOBAL LIGHT
# --------------------------------------------------
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(1200)
sun_xf = UsdGeom.Xformable(sun.GetPrim())
sun_xf.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

# --------------------------------------------------
# TWO SIMULATIONS
# --------------------------------------------------
make_xform("/World/SimLeft", translate=(-20, 0, 0))
make_xform("/World/SimRight", translate=(20, 0, 0))

build_simulation("/World/SimLeft")
build_simulation("/World/SimRight")

# --------------------------------------------------
# OVERVIEW CAMERA TO SEE BOTH
# --------------------------------------------------
camera = UsdGeom.Camera.Define(stage, "/World/Camera")
camera_xf = UsdGeom.Xformable(camera.GetPrim())
camera_xf.AddTranslateOp().Set(Gf.Vec3d(0, 10, 42))
camera_xf.AddRotateXYZOp().Set(Gf.Vec3f(-12, 0, 0))

# --------------------------------------------------
# SAVE
# --------------------------------------------------
stage.GetRootLayer().Save()
print("Saved:", output_usd)