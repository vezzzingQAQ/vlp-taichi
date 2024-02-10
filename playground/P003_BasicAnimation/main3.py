import taichi as ti

ti.init(arch=ti.cpu)

n = 512

pixels = ti.Vector.field(3, ti.f32, shape=(n, n))


@ti.func
def frac(x):
    return x-ti.floor(x)


@ti.kernel
def paint(t: ti.f32):
    for i_, j_ in pixels:

        c = 0.0
        levels = 7
        for k in range(levels):
            block_size = 1 * 2 ** k

            i = i_ + t
            j = j_ + t

            p = i % block_size / block_size
            q = j % block_size / block_size
            brightness = (0.9 - ti.Vector([p - 0.5, q - 0.5]).norm()) * 2

            i = i // block_size
            j = j // block_size

            weight = 0.5 ** (levels - k - 1) * brightness

            c += frac(ti.sin(float(i * 8 + j * 42 + t * 0.5e-3))
                      * 168) * weight
            c *= 0.8
        pixels[i_, j_] = ti.Vector([c, c * 0.8, c * levels * 0.3])


gui = ti.GUI('shaderToy', (n, n))
t = 0.0
while True:
    t += 0.2
    paint(t)
    gui.set_image(pixels)
    gui.show()
