import sys
from PIL import Image
from glob import glob
from IPython import embed

fig2d_paths = glob("2d/*.jpg")
fig3d_paths = glob("3d/*.jpg")
fig2d_paths.sort()
fig3d_paths.sort()
fig_paths = zip(fig2d_paths, fig3d_paths)

count = 0
for paths in fig_paths:
    figs = list(map(Image.open, paths))
    widths, heights = zip(*(i.size for i in figs))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))
    new_im.paste((255,255,255),[0,0,new_im.size[0],new_im.size[1]])
    x_offset = 0
    y_offset = 0
    for fig in figs:
        new_im.paste(fig, (x_offset, y_offset))
        x_offset += fig.size[0]
        y_offset = abs(heights[0]-heights[1])//2

    new_im.save(f"fig{str(count).zfill(4)}.jpg")
    count += 1
