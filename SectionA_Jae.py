import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import rotate, resize

img = io.imread('Group_6.png')

vcols = [0, 222, 444, 666, 888, 1110, 1332, 1554, 1776, 1998]
hrows = [0, 166, 332, 498, 664, 830, 996]

def get_letter(row, col):
    y1, y2 = hrows[row], hrows[row+1]
    x1, x2 = vcols[col], vcols[col+1]
    return img[y1+8:y2-8, x1+8:x2-8]   #trim border lines

def rotate_letter(cell, angle):
    r = rotate(cell, angle, resize=False, cval=1.0, mode='constant')
    return (np.clip(r, 0, 1) * 255).astype(np.uint8)

def tight_crop(cell, pad=8, threshold=180):
    gray = cell.mean(axis=2)
    dark = gray < threshold
    rows = np.where(dark.any(axis=1))[0]
    cols = np.where(dark.any(axis=0))[0]
    if len(rows) == 0: return cell
    return cell[rows[0]-pad : rows[-1]+pad,
                cols[0]-pad : cols[-1]+pad]

def place_on_canvas(cell, canvas_h=200, canvas_w=100, threshold=180):
    # Scale to fit canvas
    for max_dim, axis in [(canvas_w-10, 1), (canvas_h-10, 0)]:
        if cell.shape[axis] > max_dim:
            scale = max_dim / cell.shape[axis]
            nh = max(1, int(cell.shape[0]*scale))
            nw = max(1, int(cell.shape[1]*scale))
            cell = (resize(cell,(nh,nw),anti_aliasing=True)*255).astype(np.uint8)

    gray = cell.mean(axis=2)
    rows = np.where((gray < threshold).any(axis=1))[0]
    bottom = rows[-1] if len(rows) else cell.shape[0]-1

    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    BASELINE = canvas_h - 20 #align bottom for all letters
    y_off = BASELINE - bottom
    x_off = (canvas_w - cell.shape[1]) // 2
    y1 = max(0, y_off);  y2 = min(canvas_h, y_off + cell.shape[0])
    x1 = max(0, x_off);  x2 = min(canvas_w, x_off + cell.shape[1])
    canvas[y1:y2, x1:x2] = cell[y1-y_off : y2-y_off, x1-x_off : x2-x_off]
    return canvas

def process(row, col, angle):
    return place_on_canvas(tight_crop(rotate_letter(get_letter(row, col), angle)))

#first name
W_img = process(2, 4, angle=337.5)
e_img = process(5, 2, angle=-20)
i_img = process(1, 0, angle= 10)
q_img = process(3, 7, angle= 5)

#last name
L_img = process(5, 1, angle=28)
o_img = process(1, 1, angle=210)

#combine first and last name into 1 subplot
fig, axes = plt.subplots(1, 8, figsize=(14, 3))
fig.suptitle('My full name is Weiqi Loo, I am slicing the image from Group_6.png to extract my name.', fontsize=14)

for ax, letter in zip(axes, [W_img, e_img, i_img, q_img, i_img, L_img, o_img, o_img]):
    ax.imshow(letter)
    ax.axis('off')

plt.subplots_adjust(wspace=0.01)
plt.tight_layout()
plt.show()
