import svgwrite
import numpy as np

from .path_generator import CHANNEL_COLORS

_PLOT_ORDER = ['Y', 'C', 'M', 'K']  # lightest → darkest


def export_multi_channel_svg(paths, width, height, filename,
                              pen_width=1.0, pen_colors=None):
    """
    Export all channel paths to a layered SVG file.

    paths       : dict  channel_name -> Nx2 numpy array
    width/height: image dimensions in pixels
    filename    : output .svg path
    pen_width   : stroke width in SVG user units (pixels)
    pen_colors  : optional override dict channel_name -> '#RRGGBB'
    """
    colors = dict(CHANNEL_COLORS)
    if pen_colors:
        colors.update(pen_colors)

    dwg = svgwrite.Drawing(
        filename,
        size=(f'{width}px', f'{height}px'),
        viewBox=f'0 0 {width} {height}',
    )

    for name in _PLOT_ORDER:
        pts = paths.get(name)
        if pts is None or len(pts) == 0:
            continue

        group = dwg.add(dwg.g(
            id=f'layer_{name}',
            stroke=colors[name],
            fill='none',
            stroke_width=pen_width,
            stroke_linecap='round',
            stroke_linejoin='round',
        ))

        point_list = [(float(p[0]), float(p[1])) for p in pts]
        group.add(dwg.polyline(point_list))

    dwg.save()
    total_pts = sum(len(p) for p in paths.values() if len(p) > 0)
    return total_pts
