import taichi as ti
ti.init(arch=ti.cpu)

res_x = 64*12
res_y = 64*12
pixels = ti.Vector.field(3, ti.f32, shape=(res_x, res_y))


@ti.func
def circle(pos, center, radius: ti.f32):
    r = (pos-center).norm()
    val = 0.0
    if r < radius:
        val = 1.0
    return val


@ti.kernel
def render(t: ti.f32):
    # draw something on your canvas
    for i_, j_ in pixels:
        color = ti.Vector([0.0, 0.0, 0.0])

        tile_size = 4

        for k in range(7):
            i = i_-ti.floor(i_/tile_size)*tile_size
            j = j_-ti.floor(j_/tile_size)*tile_size
            
            pos = ti.Vector([i, j])
            center = ti.Vector([tile_size/2.0, tile_size/2.0])

            radius = tile_size/2.0
            c = circle(pos, center, radius)
            color += c*ti.Vector([1.0, 1.0, 1.0])

            color /= 3.0
            tile_size *= 2

        pixels[i_, j_] = color


@ti.func
def cal(x: ti.i32, y: ti.i32, t: ti.f32):
    z = 0.5*ti.sin(float(x*y*0.0001+t))+0.5
    return z


gui = ti.GUI('canvas', res=(res_x, res_y))

for i in range(100000):
    render(i*0.1)
    gui.set_image(pixels)
    gui.show()
