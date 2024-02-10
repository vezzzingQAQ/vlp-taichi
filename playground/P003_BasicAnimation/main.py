import taichi as ti
ti.init(arch=ti.cpu)

res_x = 912
res_y = 912
pixels = ti.Vector.field(3, ti.f32, shape=(res_x, res_y))


@ti.kernel
def render(t: ti.f32):
    # draw something on your canvas
    for i, j in pixels:
        # r=0.5*ti.sin(float(i)/res_x)+0.5
        # g=0.5*ti.sin(float(j)/res_y+2)+0.5
        # b=0.5*ti.sin(float(i)/res_x+4)+0.5
        z = cal(i-res_x/2, j-res_y/2, t)
        color = ti.Vector([z, z, z])
        pixels[i, j] = color


@ti.func
def cal(x: ti.i32, y: ti.i32, t: ti.f32):
    z = 0.5*ti.sin(float(x*y*0.0001+t))+0.5
    return z


gui = ti.GUI('canvas', res=(res_x, res_y))

for i in range(100000):
    render(i*0.1)
    gui.set_image(pixels)
    gui.show()
