import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import math
import sys

# ----------------------- Configurações Globais -----------------------
WIDTH, HEIGHT = 1920, 1080
BASE_ITER = 500

# ----------------------- Kernel CUDA com smooth coloring e float64 -----------------------
@cuda.jit
def mandelbrot_gpu_smooth(img, re_start, re_end, im_start, im_end, max_iter):
    y, x = cuda.grid(2)
    height, width = img.shape
    if y < height and x < width:
        real = re_start + x / width * (re_end - re_start)
        imag = im_start + y / height * (im_end - im_start)
        c = complex(real, imag)
        z = complex(0.0, 0.0)
        iter_count = 0
        while (z.real*z.real + z.imag*z.imag) <= 4.0 and iter_count < max_iter:
            z = z*z + c
            iter_count += 1
        if iter_count < max_iter:
            log_zn = math.log(z.real*z.real + z.imag*z.imag) / 2.0
            nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
            img[y, x] = iter_count + 1 - nu
        else:
            img[y, x] = 0.0

# ----------------------- Função para gerar imagem -----------------------
def generate_mandelbrot(center_x, center_y, zoom, width, height, max_iter):
    scale = 1.0 / zoom
    range_real = 3.5 * scale
    range_imag = 2.0 * scale
    re_start = center_x - range_real / 2.0
    re_end = center_x + range_real / 2.0
    im_start = center_y - range_imag / 2.0
    im_end = center_y + range_imag / 2.0

    img_device = cuda.device_array((height, width), dtype=np.float64)
    threadsperblock = (16, 16)
    blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]

    mandelbrot_gpu_smooth[(blockspergrid_y, blockspergrid_x), threadsperblock](
        img_device, re_start, re_end, im_start, im_end, max_iter
    )
    img_host = img_device.copy_to_host()
    norm_img = np.log(img_host + 1.0)
    norm_img /= norm_img.max()
    return norm_img

# ----------------------- Zoom Interativo -----------------------
def interactive_zoom():
    fig, ax = plt.subplots()
    zoom = 1.0
    center_x, center_y = -0.5, 0.0

    def draw():
        ax.clear()
        iter_count = int(BASE_ITER * math.log(zoom + 1))
        img = generate_mandelbrot(center_x, center_y, zoom, WIDTH, HEIGHT, iter_count)
        ax.imshow(img, cmap='inferno', extent=[0, WIDTH, HEIGHT, 0])
        ax.set_title(f"Zoom: {zoom:.2e} | Iter: {iter_count} | Centro: ({center_x:.10f}, {center_y:.10f})")
        plt.draw()

    def onclick(event):
        nonlocal center_x, center_y, zoom
        if event.inaxes != ax:
            return
        real = center_x + (event.xdata - WIDTH / 2) * (3.5 / zoom / WIDTH)
        imag = center_y + (event.ydata - HEIGHT / 2) * (2.0 / zoom / HEIGHT)
        center_x, center_y = real, imag
        if event.button == 1:  # Botão esquerdo do mouse
            zoom *= 2.0
        elif event.button == 3:  # Botão direito do mouse
            zoom /= 2.0
        draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    draw()
    plt.show()

if __name__ == "__main__":
    interactive_zoom()