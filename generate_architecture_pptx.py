"""
Generate 100% Native Editable PowerPoint (.pptx) Architecture Diagrams for OrthoSVM-HHG.
Strictly follows the academic visual style of the reference Information Sciences diagram:
- Crisp modular black-bordered containers with dashed sub-panel dividers & bold-italic serif titles
- True OpenXML mathematical subscripts/superscripts (c_i, p_i, R_1, f_r^{raw}, Π_⊥, ŷ_i)
- Rich native editable vector mini-icons (grouped shapes: judicial gavels, Hawkes decay bars,
  coin bag, corporate building shield, lifecycle clock, dual database cylinders, 5-fold OOF matrix,
  Mercer partition-tree + kernel landmark icon, Riemannian orthogonal projection geometry,
  Vapnik max-margin hyperplane slab, and Saerens-EM prior shift bars)
- Tilted 3D isometric embedding prisms (native MSO_SHAPE.CUBE) in academic pastel palette
- Upright regular pentagons, sleek curved self-loop arcs, bracketed structural pathways, and entity legend.
"""

import os
import re
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.ns import qn
from lxml import etree


# =============================================================================
# ACADEMIC PASTEL & ACCENT COLOR PALETTE (MATCHING REFERENCE STYLE)
# =============================================================================
C = {
    'black':      RGBColor(28, 28, 32),
    'dark_gray':  RGBColor(65, 68, 76),
    'mid_gray':   RGBColor(130, 135, 145),
    'light_gray': RGBColor(236, 238, 242),
    'panel_gray': RGBColor(246, 247, 250),
    'white':      RGBColor(255, 255, 255),

    # Entity & Embedding Pastels (Faithful to Reference Image)
    'company':    RGBColor(174, 190, 226),  # Soft Periwinkle Circle
    'person':     RGBColor(244, 177, 131),  # Soft Peach/Orange Triangle
    'area':       RGBColor(78, 154, 81),    # Forest Emerald Diamond
    'sector':     RGBColor(168, 142, 194),  # Muted Lavender Trapezoid
    'syndicate':  RGBColor(255, 230, 153),  # Warm Cream-Yellow Regular Pentagon

    # 3D Cuboid Embedding Colors
    'cub_blue':   RGBColor(115, 158, 218),
    'cub_green':  RGBColor(152, 202, 132),
    'cub_pink':   RGBColor(222, 162, 198),
    'cub_orange': RGBColor(236, 157, 108),
    'cub_teal':   RGBColor(118, 183, 178),
    'cub_purple': RGBColor(148, 130, 190),
    'cub_indigo': RGBColor(108, 104, 168),
    'cub_red':    RGBColor(214, 78, 72),
    'cub_yellow': RGBColor(245, 218, 142),
    'cub_rose':   RGBColor(234, 192, 210),

    # Accents
    'block_arr':  RGBColor(217, 98, 82),    # Coral-Red Macro Block Arrow
    'target_red': RGBColor(200, 38, 38),    # Red c_i highlight text
    'box_lav':    RGBColor(224, 220, 232),  # Muted Lavender-Grey Function Box
    'box_green':  RGBColor(222, 242, 232),  # Soft Mint Resilience Box
    'box_red':    RGBColor(252, 228, 226),  # Soft Coral Risk Box
    'box_blue':   RGBColor(225, 236, 250),  # Soft Sky Projector Box
    'dash_yel':   RGBColor(230, 185, 60),
    'dash_pur':   RGBColor(150, 110, 185),
    'dash_org':   RGBColor(225, 140, 75),
}

TOKEN_PATTERN = re.compile(r'(_\{[^}]+\}|_[A-Za-z0-9⊥π]+|\^\{[^}]+\}|\^[A-Za-z0-9⊥⊤′]+)')


# =============================================================================
# NATIVE OPENXML MATH SUBSCRIPT/SUPERSCRIPT & SHAPE HELPERS
# =============================================================================
def set_text(shape, text, size=9.0, bold=False, italic=False, color=C['black'],
             font_name='Times New Roman', align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
             margin=0.015, word_wrap=True):
    """Formats text with automatic native PowerPoint subscripts (_x / _{xy}) and superscripts (^x / ^{xy})."""
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = word_wrap
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)

    lines = text.split('\n')
    for l_idx, line in enumerate(lines):
        p = tf.paragraphs[0] if l_idx == 0 else tf.add_paragraph()
        p.alignment = align
        parts = TOKEN_PATTERN.split(line)
        for part in parts:
            if not part:
                continue
            run = p.add_run()
            baseline = None
            if part.startswith('_{') and part.endswith('}'):
                run.text = part[2:-1]
                baseline = '-28000'
            elif part.startswith('_') and len(part) > 1:
                run.text = part[1:]
                baseline = '-28000'
            elif part.startswith('^{') and part.endswith('}'):
                run.text = part[2:-1]
                baseline = '32000'
            elif part.startswith('^') and len(part) > 1:
                run.text = part[1:]
                baseline = '32000'
            else:
                run.text = part

            run.font.name = font_name
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.italic = italic
            run.font.color.rgb = color
            if baseline is not None:
                rPr = run._r.get_or_add_rPr()
                rPr.set('baseline', baseline)
    return tf


def add_label(slide, x, y, w, h, text, size=9.0, bold=False, italic=False,
              color=C['black'], font_name='Times New Roman', align=PP_ALIGN.CENTER,
              word_wrap=True):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    set_text(tb, text, size=size, bold=bold, italic=italic, color=color,
             font_name=font_name, align=align, margin=0.01, word_wrap=word_wrap)
    return tb


def add_box(slide, x, y, w, h, fill=C['white'], border=C['black'], lw=1.0,
            dash=None, rounded=False, text=None, font_size=9.0, bold=False,
            italic=False, text_color=C['black'], font_name='Times New Roman'):
    stype = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    sh = slide.shapes.add_shape(stype, Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded and len(sh.adjustments) > 0:
        sh.adjustments[0] = 0.12
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if border is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = border
        sh.line.width = Pt(lw)
        if dash is not None:
            sh.line.dash_style = dash
    if text is not None:
        set_text(sh, text, size=font_size, bold=bold, italic=italic,
                 color=text_color, font_name=font_name)
    return sh


def style_line_ends(line_obj, head=False, tail=True, arrow_type='stealth'):
    ln = line_obj._get_or_add_ln()
    for tag in ('a:headEnd', 'a:tailEnd'):
        old = ln.find(qn(tag))
        if old is not None:
            ln.remove(old)
    if head:
        he = etree.SubElement(ln, qn('a:headEnd'))
        he.set('type', arrow_type)
        he.set('w', 'med')
        he.set('len', 'med')
    if tail:
        te = etree.SubElement(ln, qn('a:tailEnd'))
        te.set('type', arrow_type)
        te.set('w', 'med')
        te.set('len', 'med')


def add_line_arrow(slide, x1, y1, x2, y2, color=C['black'], lw=1.05, dash=None,
                   head=False, tail=True, arrow_type='stealth'):
    conn = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    conn.line.color.rgb = color
    conn.line.width = Pt(lw)
    if dash is not None:
        conn.line.dash_style = dash
    style_line_ends(conn.line, head=head, tail=tail, arrow_type=arrow_type)
    return conn


def add_curved_loop_arrow(slide, x, y, w, h, rotation=0.0, color=C['black'], lw=0.95):
    """Sleek thin curved loop arrow using native MSO_SHAPE.ARC (matches Meta-Template loops)."""
    arc = slide.shapes.add_shape(MSO_SHAPE.ARC, Inches(x), Inches(y), Inches(w), Inches(h))
    if len(arc.adjustments) >= 2:
        arc.adjustments[0] = 210.0
        arc.adjustments[1] = 150.0
    arc.fill.background()
    arc.line.color.rgb = color
    arc.line.width = Pt(lw)
    arc.rotation = rotation
    style_line_ends(arc.line, head=False, tail=True, arrow_type='stealth')
    return arc


def add_cuboid(slide, x, y, w=0.25, h=0.66, color=C['cub_blue'], angle=-36.0, lw=0.85):
    cub = slide.shapes.add_shape(MSO_SHAPE.CUBE, Inches(x), Inches(y), Inches(w), Inches(h))
    if len(cub.adjustments) > 0:
        cub.adjustments[0] = 0.28
    cub.fill.solid()
    cub.fill.fore_color.rgb = color
    cub.line.color.rgb = C['black']
    cub.line.width = Pt(lw)
    cub.rotation = angle
    return cub


def add_operator_circle(slide, cx, cy, r=0.115, symbol='+', size=10.5, fill=C['white']):
    ov = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, Inches(cx - r), Inches(cy - r), Inches(2 * r), Inches(2 * r)
    )
    ov.fill.solid()
    ov.fill.fore_color.rgb = fill
    ov.line.color.rgb = C['black']
    ov.line.width = Pt(0.95)
    set_text(ov, symbol, size=size, bold=True, font_name='Cambria Math', margin=0.0, word_wrap=False)
    return ov


def add_entity_node(slide, kind, cx, cy, r=0.145, label='', is_target=False, lw=0.9):
    """Typed entity icon using upright REGULAR_PENTAGON for syndicate/env (exact reference match)."""
    mapping = {
        'company':   (MSO_SHAPE.OVAL,               C['company'],   1.02, 1.02),
        'person':    (MSO_SHAPE.ISOSCELES_TRIANGLE, C['person'],    1.18, 1.12),
        'area':      (MSO_SHAPE.DIAMOND,            C['area'],      1.16, 1.16),
        'sector':    (MSO_SHAPE.TRAPEZOID,          C['sector'],    1.12, 1.02),
        'syndicate': (MSO_SHAPE.REGULAR_PENTAGON,   C['syndicate'], 1.12, 1.10),
        'env_white': (MSO_SHAPE.REGULAR_PENTAGON,   C['white'],     1.12, 1.10),
    }
    st, col, sx, sy = mapping[kind]
    w, h = 2 * r * sx, 2 * r * sy
    sh = slide.shapes.add_shape(st, Inches(cx - w / 2), Inches(cy - h / 2), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = col
    sh.line.color.rgb = C['target_red'] if is_target else C['black']
    sh.line.width = Pt(1.25 if is_target else lw)
    if label:
        tcol = C['target_red'] if is_target else (C['white'] if kind == 'area' else C['black'])
        anch = MSO_ANCHOR.BOTTOM if kind == 'person' else MSO_ANCHOR.MIDDLE
        set_text(sh, label, size=8.0 if kind == 'person' else 8.4,
                 bold=is_target, italic=True, color=tcol,
                 font_name='Times New Roman', anchor=anch, margin=0.0, word_wrap=False)
    return sh


def add_macro_arrow(slide, x, y, w, h, direction='right'):
    shape_map = {
        'right': MSO_SHAPE.RIGHT_ARROW,
        'up':    MSO_SHAPE.UP_ARROW,
        'down':  MSO_SHAPE.DOWN_ARROW,
    }
    arr = slide.shapes.add_shape(shape_map[direction], Inches(x), Inches(y), Inches(w), Inches(h))
    arr.fill.solid()
    arr.fill.fore_color.rgb = C['block_arr']
    arr.line.color.rgb = C['black']
    arr.line.width = Pt(0.9)
    return arr


def add_stacked_cards(slide, x, y, w, h, n=3, dx=0.05, dy=0.048, fill=C['light_gray'],
                      top_text='', font_size=8.0):
    cards = []
    for i in range(n):
        sh = add_box(slide, x + i * dx, y + i * dy, w, h, fill=fill, border=C['black'], lw=0.85)
        cards.append(sh)
    set_text(cards[-1], top_text, size=font_size, bold=False, font_name='Times New Roman')
    return cards


# =============================================================================
# COMPOSITE EDITABLE VECTOR MINI-ICONS (GROUPED NATIVE SHAPES)
# =============================================================================
def add_gavel_icon(slide, x, y, scale=1.0):
    grp = slide.shapes.add_group_shape()
    b = grp.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y + 0.22 * scale),
                             Inches(0.24 * scale), Inches(0.055 * scale))
    b.fill.solid(); b.fill.fore_color.rgb = C['dark_gray']
    b.line.color.rgb = C['black']; b.line.width = Pt(0.7)
    hdl = grp.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x + 0.13 * scale), Inches(y + 0.08 * scale),
                               Inches(0.18 * scale), Inches(0.04 * scale))
    hdl.fill.solid(); hdl.fill.fore_color.rgb = C['white']
    hdl.line.color.rgb = C['black']; hdl.line.width = Pt(0.75)
    hdl.rotation = 35.0
    hd = grp.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x + 0.04 * scale), Inches(y + 0.01 * scale),
                              Inches(0.16 * scale), Inches(0.095 * scale))
    hd.fill.solid(); hd.fill.fore_color.rgb = C['light_gray']
    hd.line.color.rgb = C['black']; hd.line.width = Pt(0.85)
    hd.rotation = -52.0
    return grp


def add_coin_bag_icon(slide, x, y, scale=1.0):
    grp = slide.shapes.add_group_shape()
    pouch = grp.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y + 0.07 * scale),
                                 Inches(0.25 * scale), Inches(0.22 * scale))
    pouch.fill.solid(); pouch.fill.fore_color.rgb = RGBColor(253, 242, 208)
    pouch.line.color.rgb = C['black']; pouch.line.width = Pt(0.85)
    set_text(pouch, '¥', size=7.8 * scale, bold=True, font_name='Arial', margin=0.0, word_wrap=False)
    top = grp.shapes.add_shape(MSO_SHAPE.TRAPEZOID, Inches(x + 0.055 * scale), Inches(y + 0.01 * scale),
                               Inches(0.14 * scale), Inches(0.075 * scale))
    top.rotation = 180.0
    top.fill.solid(); top.fill.fore_color.rgb = RGBColor(244, 212, 138)
    top.line.color.rgb = C['black']; top.line.width = Pt(0.8)
    return grp


def add_corporate_shield_icon(slide, x, y, scale=1.0):
    grp = slide.shapes.add_group_shape()
    bld = grp.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y + 0.02 * scale),
                               Inches(0.20 * scale), Inches(0.26 * scale))
    bld.fill.solid(); bld.fill.fore_color.rgb = C['box_blue']
    bld.line.color.rgb = C['black']; bld.line.width = Pt(0.85)
    for r_i in range(2):
        for c_i in range(2):
            win = grp.shapes.add_shape(
                MSO_SHAPE.RECTANGLE,
                Inches(x + (0.035 + 0.08 * c_i) * scale),
                Inches(y + (0.06 + 0.08 * r_i) * scale),
                Inches(0.045 * scale), Inches(0.045 * scale)
            )
            win.fill.solid(); win.fill.fore_color.rgb = C['white']
            win.line.color.rgb = C['black']; win.line.width = Pt(0.6)
    shd = grp.shapes.add_shape(MSO_SHAPE.REGULAR_PENTAGON, Inches(x + 0.13 * scale), Inches(y + 0.13 * scale),
                               Inches(0.14 * scale), Inches(0.15 * scale))
    shd.rotation = 180.0
    shd.fill.solid(); shd.fill.fore_color.rgb = C['cub_green']
    shd.line.color.rgb = C['black']; shd.line.width = Pt(0.75)
    return grp


def add_tenure_clock_icon(slide, x, y, scale=1.0):
    grp = slide.shapes.add_group_shape()
    clk = grp.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y + 0.03 * scale),
                               Inches(0.24 * scale), Inches(0.24 * scale))
    clk.fill.solid(); clk.fill.fore_color.rgb = C['box_green']
    clk.line.color.rgb = C['black']; clk.line.width = Pt(0.85)
    h1 = grp.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x + 0.115 * scale), Inches(y + 0.07 * scale),
                              Inches(0.015 * scale), Inches(0.085 * scale))
    h1.fill.solid(); h1.fill.fore_color.rgb = C['black']; h1.line.fill.background()
    h2 = grp.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x + 0.115 * scale), Inches(y + 0.145 * scale),
                              Inches(0.075 * scale), Inches(0.015 * scale))
    h2.fill.solid(); h2.fill.fore_color.rgb = C['target_red']; h2.line.fill.background()
    return grp


def add_db_cylinders_icon(slide, x, y, scale=1.0):
    grp = slide.shapes.add_group_shape()
    c1 = grp.shapes.add_shape(MSO_SHAPE.CAN, Inches(x), Inches(y),
                              Inches(0.24 * scale), Inches(0.28 * scale))
    c1.fill.solid(); c1.fill.fore_color.rgb = C['white']
    c1.line.color.rgb = C['black']; c1.line.width = Pt(0.85)
    c2 = grp.shapes.add_shape(MSO_SHAPE.CAN, Inches(x + 0.16 * scale), Inches(y + 0.07 * scale),
                              Inches(0.22 * scale), Inches(0.23 * scale))
    c2.fill.solid(); c2.fill.fore_color.rgb = C['light_gray']
    c2.line.color.rgb = C['black']; c2.line.width = Pt(0.85)
    return grp


def add_hawkes_curve_icon(slide, x, y, w=0.46, h=0.26):
    add_line_arrow(slide, x, y + h, x + w, y + h, lw=0.75, tail=True)
    add_line_arrow(slide, x, y + h, x, y, lw=0.75, tail=True)
    heights = [0.85, 0.52, 0.30, 0.16]
    cols = [C['cub_red'], C['cub_orange'], C['cub_yellow'], C['cub_green']]
    bw = (w - 0.08) / 4.8
    for i, (hr, col) in enumerate(zip(heights, cols)):
        bh = h * hr
        bx = x + 0.04 + i * (bw + 0.018)
        add_box(slide, bx, y + h - bh, bw, bh, fill=col, border=C['black'], lw=0.55)


def add_kfold_matrix_icon(slide, x, y, cell=0.088, gap=0.016):
    """Editable 5x5 Cross-Fitting OOF fold grid icon (diagonal held-out folds in coral red)."""
    grp = slide.shapes.add_group_shape()
    for r in range(5):
        for c in range(5):
            bx = x + c * (cell + gap)
            by = y + r * (cell + gap)
            sh = grp.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(bx), Inches(by), Inches(cell), Inches(cell))
            sh.fill.solid()
            sh.fill.fore_color.rgb = C['block_arr'] if r == c else C['box_blue']
            sh.line.color.rgb = C['black']
            sh.line.width = Pt(0.55)
    return grp


def add_mercer_tree_kernel_icon(slide, x, y):
    """Editable Partition-Tree + Kernel Landmark structural diagram."""
    # Mini tree structure on left (x..x+0.56)
    add_entity_node(slide, 'company', x + 0.28, y + 0.08, r=0.065, lw=0.7)
    add_entity_node(slide, 'area',    x + 0.13, y + 0.26, r=0.060, lw=0.7)
    add_entity_node(slide, 'sector',  x + 0.43, y + 0.26, r=0.060, lw=0.7)
    add_line_arrow(slide, x + 0.24, y + 0.13, x + 0.15, y + 0.21, lw=0.75, tail=False)
    add_line_arrow(slide, x + 0.32, y + 0.13, x + 0.41, y + 0.21, lw=0.75, tail=False)
    for li, lx in enumerate([0.05, 0.19, 0.35, 0.49]):
        add_box(slide, x + lx, y + 0.40, 0.095, 0.095,
                fill=C['cub_teal'] if li in (1, 3) else C['light_gray'], border=C['black'], lw=0.6)
    add_line_arrow(slide, x + 0.11, y + 0.31, x + 0.09, y + 0.40, lw=0.65, tail=False)
    add_line_arrow(slide, x + 0.15, y + 0.31, x + 0.23, y + 0.40, lw=0.65, tail=False)
    add_line_arrow(slide, x + 0.41, y + 0.31, x + 0.39, y + 0.40, lw=0.65, tail=False)
    add_line_arrow(slide, x + 0.45, y + 0.31, x + 0.53, y + 0.40, lw=0.65, tail=False)


def add_ortho_projector_diagram(slide, x, y, w=0.80, h=0.98):
    """Miniature geometric icon showing Riemannian Orthogonal Complement Projection Π_{S_nr^⊥}."""
    add_box(slide, x, y, w, h, fill=C['box_blue'], border=C['black'], lw=0.85, rounded=True)
    ox, oy = x + 0.15, y + h - 0.20
    # Horizontal green resilience basis vector f_nr
    add_line_arrow(slide, ox, oy, ox + 0.54, oy, color=C['area'], lw=1.45, tail=True)
    # Vertical navy orthogonal net-risk vector f_r^⊥
    add_line_arrow(slide, ox, oy, ox, oy - 0.52, color=C['cub_indigo'], lw=1.45, tail=True)
    # Tilted raw risk vector f_r^raw
    add_line_arrow(slide, ox, oy, ox + 0.44, oy - 0.52, color=C['cub_red'], lw=1.1,
                   dash=MSO_LINE_DASH_STYLE.DASH, tail=True)
    # Projection drop line
    add_line_arrow(slide, ox + 0.44, oy - 0.52, ox, oy - 0.52, color=C['dark_gray'],
                   lw=0.75, dash=MSO_LINE_DASH_STYLE.ROUND_DOT, tail=True)
    add_box(slide, ox, oy - 0.09, 0.09, 0.09, fill=None, border=C['black'], lw=0.7)
    add_label(slide, ox + 0.20, oy + 0.01, 0.44, 0.16, 'S_{nr}', size=7.2, bold=True,
              italic=True, color=C['area'], word_wrap=False)
    add_label(slide, x + 0.04, y + 0.02, w - 0.08, 0.18, 'Π_⊥ f_r', size=7.4, bold=True,
              italic=True, color=C['cub_indigo'], word_wrap=False)


def add_svm_hyperplane_icon(slide, x, y, w=0.84, h=0.86, show_caption=True):
    """Miniature Vapnik Maximum-Margin Separating Hyperplane icon."""
    add_box(slide, x, y, w, h, fill=C['white'], border=C['black'], lw=0.85, rounded=True)
    pad_b = 0.21 if show_caption else 0.10
    add_line_arrow(slide, x + 0.12, y + h - pad_b, x + w - 0.12, y + 0.09,
                   color=C['black'], lw=1.35, tail=False)
    add_line_arrow(slide, x + 0.06, y + h - pad_b - 0.08, x + w - 0.20, y + 0.06,
                   color=C['cub_red'], lw=0.75, dash=MSO_LINE_DASH_STYLE.DASH, tail=False)
    add_line_arrow(slide, x + 0.20, y + h - pad_b + 0.03, x + w - 0.06, y + 0.17,
                   color=C['area'], lw=0.75, dash=MSO_LINE_DASH_STYLE.DASH, tail=False)
    for px, py in [(x + 0.16, y + 0.13), (x + 0.30, y + 0.10), (x + 0.14, y + 0.27)]:
        d = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(px), Inches(py), Inches(0.08), Inches(0.08))
        d.fill.solid(); d.fill.fore_color.rgb = C['cub_red']; d.line.color.rgb = C['black']; d.line.width = Pt(0.5)
    for nx, ny in [(x + 0.54, y + 0.34), (x + 0.64, y + 0.23), (x + 0.46, y + 0.44)]:
        d = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(nx), Inches(ny), Inches(0.08), Inches(0.08))
        d.fill.solid(); d.fill.fore_color.rgb = C['cub_green']; d.line.color.rgb = C['black']; d.line.width = Pt(0.5)
    if show_caption:
        add_label(slide, x + 0.02, y + h - 0.19, w - 0.04, 0.17, 'Margin d(x)',
                  size=6.9, bold=True, word_wrap=False)


# =============================================================================
# SLIDE 1: OVERALL ARCHITECTURE OF OrthoSVM-HHG (REFERENCE BPHD STYLE)
# =============================================================================
def build_slide1_overall_architecture(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # -------------------------------------------------------------------------
    # 1. TOP-LEFT MODULE: ENDOGENOUS HAWKES & LIFECYCLE MODULE (MH-CLE)
    # -------------------------------------------------------------------------
    add_box(slide, 0.18, 0.18, 6.08, 2.42, fill=C['white'], border=C['black'], lw=1.15)
    add_entity_node(slide, 'company', 0.45, 1.32, r=0.145, label='c_i', is_target=True)

    br = slide.shapes.add_shape(MSO_SHAPE.LEFT_BRACE, Inches(0.66), Inches(0.42), Inches(0.20), Inches(1.78))
    br.line.color.rgb = C['black']; br.line.width = Pt(1.1); br.fill.background()

    # Upper Row: 3 Vector Gavels + Hawkes Exponential Decay Icon
    add_gavel_icon(slide, 0.94, 0.44, scale=0.92)
    add_gavel_icon(slide, 1.30, 0.44, scale=0.92)
    add_gavel_icon(slide, 1.66, 0.44, scale=0.92)
    add_hawkes_curve_icon(slide, 2.06, 0.43, w=0.48, h=0.27)
    add_label(slide, 0.88, 0.75, 1.72, 0.22, 'Hawkes Lawsuit Sequence', size=8.3)

    # Lower Row: Paid-in Coin Bag + Corporate Shield + Lifecycle Clock
    add_coin_bag_icon(slide, 0.95, 1.32, scale=0.98)
    add_corporate_shield_icon(slide, 1.46, 1.32, scale=0.98)
    add_tenure_clock_icon(slide, 2.02, 1.32, scale=0.98)
    add_label(slide, 2.30, 1.38, 0.35, 0.20, '......', size=9.5, bold=True, word_wrap=False)
    add_label(slide, 0.86, 1.67, 1.76, 0.22, 'Capital & Lifecycle Profile', size=8.3)

    add_label(slide, 0.65, 2.18, 2.05, 0.28, 'Endogenous Records', size=10.2, bold=True)
    add_macro_arrow(slide, 2.68, 1.23, 0.34, 0.30, 'right')

    # Inner Encoding Sub-Frame with 3 Tilted 3D Cuboids
    add_box(slide, 3.10, 0.34, 3.02, 2.10, fill=C['white'], border=C['black'], lw=0.95)
    add_cuboid(slide, 3.38, 0.56, w=0.25, h=0.68, color=C['cub_blue'], angle=-36)
    add_label(slide, 3.15, 1.34, 0.78, 0.36, 'Conditional\nLifecycle Emb.', size=7.8)

    add_operator_circle(slide, 4.02, 0.98, r=0.105, symbol='+')

    add_cuboid(slide, 4.35, 0.56, w=0.25, h=0.68, color=C['cub_green'], angle=-36)
    add_label(slide, 4.10, 1.34, 0.80, 0.36, 'Hawkes Decay\nLawsuit Emb.', size=7.8)

    add_line_arrow(slide, 4.78, 0.98, 5.14, 0.98, lw=1.1, tail=True)

    add_cuboid(slide, 5.36, 0.56, w=0.25, h=0.68, color=C['cub_pink'], angle=-36)
    add_label(slide, 5.06, 1.34, 0.94, 0.36, 'Endogenous\nProfile Emb.', size=7.8)

    add_label(slide, 3.16, 2.06, 2.90, 0.30, 'Temporal Hawkes & Lifecycle Module (MH-CLE)',
              size=9.1, bold=True, italic=True)

    # -------------------------------------------------------------------------
    # 2. TOP-CENTER/RIGHT: HILBERT-ORTHOGONAL DUAL-EXPERT SVM MODULE (HO-SVDE)
    # -------------------------------------------------------------------------
    add_macro_arrow(slide, 6.30, 1.23, 0.34, 0.32, 'right')
    add_box(slide, 6.68, 0.18, 7.46, 2.42, fill=C['white'], border=C['black'], lw=1.15)

    stack_cols = [C['cub_pink'], C['cub_orange'], C['cub_teal'], C['cub_indigo']]
    for idx, col in enumerate(stack_cols):
        cy = 0.30 + idx * 0.44
        add_cuboid(slide, 6.94, cy, w=0.21, h=0.52, color=col, angle=-42)
        add_line_arrow(slide, 7.22, cy + 0.25, 7.50, cy + 0.25, lw=0.9, tail=True)

    agg_bar = add_box(slide, 7.52, 0.32, 0.38, 1.88, fill=C['box_lav'], border=C['black'], lw=0.9)
    set_text(agg_bar, 'Multi-View\nRKHS Fusion', size=7.6)

    add_line_arrow(slide, 7.90, 1.22, 8.15, 1.22, lw=1.05, tail=True)

    add_cuboid(slide, 8.28, 0.80, w=0.24, h=0.62, color=C['cub_red'], angle=-36)
    add_label(slide, 8.04, 1.50, 0.78, 0.36, 'Canonical RKHS\nCoords z(x)', size=7.6)

    add_line_arrow(slide, 8.64, 1.02, 8.94, 0.70, lw=0.95, tail=True)
    add_line_arrow(slide, 8.64, 1.32, 8.94, 1.68, lw=0.95, tail=True)

    # Gating Deck Frame (with clean spacing before label)
    add_box(slide, 8.96, 0.26, 1.48, 0.98, fill=C['panel_gray'], border=C['dark_gray'], lw=0.75)
    add_stacked_cards(slide, 9.04, 0.32, 1.22, 0.46, n=3, dx=0.05, dy=0.042,
                      fill=C['light_gray'], top_text='Dual Gates g_r, g_{nr}', font_size=7.6)
    add_label(slide, 8.98, 1.01, 1.44, 0.22, 'Gating Networks', size=7.7, bold=True)

    # Perspective Operator Deck Frame
    add_box(slide, 8.96, 1.32, 1.48, 0.98, fill=C['panel_gray'], border=C['dark_gray'], lw=0.75)
    add_stacked_cards(slide, 9.04, 1.38, 1.22, 0.46, n=3, dx=0.05, dy=0.042,
                      fill=C['light_gray'], top_text='Hilbert Basis Ψ_1..Ψ_K', font_size=7.6)
    add_label(slide, 8.98, 2.07, 1.44, 0.22, 'Riemannian Basis', size=7.7, bold=True)

    add_operator_circle(slide, 10.68, 0.72, r=0.10, symbol='⊗', size=9.5)
    add_operator_circle(slide, 10.68, 1.72, r=0.10, symbol='⊗', size=9.5)
    add_line_arrow(slide, 10.44, 0.72, 10.58, 0.72, lw=0.9, tail=True)
    add_line_arrow(slide, 10.44, 1.72, 10.58, 1.72, lw=0.9, tail=True)
    add_line_arrow(slide, 10.44, 1.60, 10.64, 0.82, lw=0.8, tail=True)
    add_line_arrow(slide, 10.44, 0.84, 10.64, 1.62, lw=0.8, tail=True)

    # Dual-Decision Subspace Sub-Frame
    add_box(slide, 10.90, 0.26, 1.98, 2.02, fill=C['white'], border=C['black'], lw=0.9)
    add_box(slide, 10.98, 0.34, 0.92, 0.48, fill=C['box_red'], border=C['black'], lw=0.85,
            text='Risk Expert\nf_r^{raw}', font_size=7.6, bold=True)
    add_box(slide, 10.98, 1.44, 0.92, 0.48, fill=C['box_green'], border=C['black'], lw=0.85,
            text='Resilience Exp.\nf_{nr} ∈ S_{nr}', font_size=7.6, bold=True)
    add_line_arrow(slide, 10.78, 0.72, 10.98, 0.58, lw=0.9, tail=True)
    add_line_arrow(slide, 10.78, 1.72, 10.98, 1.68, lw=0.9, tail=True)

    add_ortho_projector_diagram(slide, 12.00, 0.38, w=0.80, h=1.02)
    add_line_arrow(slide, 11.90, 0.58, 12.00, 0.70, lw=0.9, tail=True)
    add_line_arrow(slide, 11.90, 1.68, 12.40, 1.40, lw=0.85, dash=MSO_LINE_DASH_STYLE.DASH, tail=True)
    add_label(slide, 10.94, 2.00, 1.90, 0.24, 'Hilbert Orthogonal OSP', size=7.8, bold=True)

    # Vapnik Max-Margin Hyperplane + Saerens-EM Calibration Output
    add_svm_hyperplane_icon(slide, 13.00, 0.28, w=0.84, h=0.90, show_caption=True)
    add_line_arrow(slide, 12.80, 0.74, 13.00, 0.64, lw=0.95, tail=True)
    add_line_arrow(slide, 11.90, 1.68, 13.00, 0.96, lw=0.85, tail=True)

    add_box(slide, 13.00, 1.32, 0.84, 0.56, fill=RGBColor(254, 245, 224), border=C['black'],
            lw=0.85, rounded=True, text='Saerens-EM\nPrior π̂_t', font_size=7.5, bold=True)
    add_line_arrow(slide, 13.42, 1.18, 13.42, 1.32, lw=0.9, tail=True)
    add_line_arrow(slide, 13.84, 1.60, 14.08, 1.60, lw=1.05, tail=True)
    add_label(slide, 13.82, 1.26, 0.34, 0.28, 'ŷ_i', size=9.8, bold=True,
              italic=True, color=C['target_red'], word_wrap=False)

    add_label(slide, 6.80, 2.31, 7.20, 0.26,
              'Hilbert-Orthogonal Dual-Expert SVM & Macro-Calibration Module (HO-SVDE)',
              size=9.4, bold=True, italic=True)

    # -------------------------------------------------------------------------
    # 3. TOP-RIGHT: TYPED ENTITY LEGEND BOX (UPRIGHT REGULAR PENTAGON)
    # -------------------------------------------------------------------------
    add_box(slide, 14.24, 0.18, 1.58, 2.42, fill=C['white'], border=C['black'], lw=1.15)
    legend_items = [
        ('company',   'company'),
        ('person',    'person'),
        ('area',      'area (city/prov)'),
        ('sector',    'sector (ind)'),
        ('syndicate', 'syndicate / env'),
    ]
    for idx, (kind, name) in enumerate(legend_items):
        ly = 0.45 + idx * 0.44
        add_entity_node(slide, kind, 14.52, ly, r=0.135)
        add_label(slide, 14.72, ly - 0.14, 1.06, 0.28, name, size=8.6, align=PP_ALIGN.LEFT)

    # -------------------------------------------------------------------------
    # 4. BOTTOM-LEFT: MULTIPLEX ENTERPRISE ECOSYSTEM (3 STRUCTURAL DIAGRAMS)
    # -------------------------------------------------------------------------
    add_box(slide, 0.18, 2.82, 3.66, 4.58, fill=C['white'], border=C['black'], lw=1.15)

    # (4a) Weighted Multiplex Heterogeneous Graph
    add_label(slide, 0.26, 3.35, 1.35, 0.48, 'Weighted\nHeterogeneous\nGraph', size=9.6, bold=True)
    g_nodes = {
        'ci': (2.62, 3.68, 'company', 'c_i', True),
        'cp': (3.12, 3.12, 'company', 'c_p', False),
        'pi': (2.42, 3.06, 'person',  'p_i', False),
        'si': (1.96, 3.56, 'sector',  's_i', False),
        'ai': (3.34, 3.66, 'area',    'a_i', False),
        'cm': (2.18, 4.14, 'company', 'c_m', False),
        'ck': (2.96, 4.18, 'company', 'c_k', False),
    }
    for u, v, lw_e in [('cp', 'ci', 2.1), ('pi', 'ci', 1.0), ('si', 'ci', 1.0),
                       ('ai', 'ci', 1.0), ('cm', 'ci', 1.0), ('ck', 'ci', 1.0),
                       ('pi', 'cp', 0.8), ('cm', 'si', 0.8), ('ck', 'ai', 0.8)]:
        add_line_arrow(slide, g_nodes[u][0], g_nodes[u][1], g_nodes[v][0], g_nodes[v][1],
                       lw=lw_e, tail=False)
    for k, (nx, ny, kind, lbl, ist) in g_nodes.items():
        add_entity_node(slide, kind, nx, ny, r=0.135, label=lbl, is_target=ist)

    # (4b) Governance & Meta-Relation Schema Network (with sleek thin curved self-loops!)
    add_label(slide, 0.26, 4.86, 1.35, 0.48, 'Governance &\nMeta-Schema\nNetwork', size=9.6, bold=True)
    add_entity_node(slide, 'company',   2.62, 5.12, r=0.165)
    add_entity_node(slide, 'area',      2.00, 4.70, r=0.125)
    add_entity_node(slide, 'sector',    2.00, 5.52, r=0.125)
    add_entity_node(slide, 'person',    3.32, 5.12, r=0.135)
    add_entity_node(slide, 'syndicate', 2.62, 4.50, r=0.120)
    add_line_arrow(slide, 2.14, 4.78, 2.46, 5.00, lw=0.9, head=True, tail=True)
    add_line_arrow(slide, 2.14, 5.44, 2.46, 5.22, lw=0.9, head=True, tail=True)
    add_line_arrow(slide, 2.80, 5.06, 3.14, 5.06, lw=0.9, tail=True)
    add_line_arrow(slide, 3.14, 5.18, 2.80, 5.18, lw=0.9, tail=True)
    add_line_arrow(slide, 2.62, 4.66, 2.62, 4.93, lw=0.9, head=True, tail=True)
    # Sleek thin curved self-loops (Invest_CC & Branch) matching reference diagram!
    add_curved_loop_arrow(slide, 2.74, 4.66, 0.34, 0.32, rotation=-25.0)
    add_curved_loop_arrow(slide, 2.45, 5.26, 0.36, 0.32, rotation=160.0)

    # (4c) Tri-View Hierarchical Hypergraph Schematic
    add_label(slide, 0.26, 6.32, 1.35, 0.52, 'Tri-View\nHierarchical\nHypergraph', size=9.6, bold=True)
    add_box(slide, 2.06, 5.88, 1.28, 0.48, fill=RGBColor(255, 246, 232),
            border=C['dash_org'], lw=0.95, dash=MSO_LINE_DASH_STYLE.DASH, rounded=True)
    add_entity_node(slide, 'syndicate', 2.34, 6.12, r=0.125, label='ε')
    add_entity_node(slide, 'env_white', 3.04, 6.12, r=0.125, label="ε'")
    add_line_arrow(slide, 2.50, 6.12, 2.88, 6.12, lw=0.9, head=True, tail=True)

    ov1 = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(1.88), Inches(6.58), Inches(1.06), Inches(0.68))
    ov1.fill.solid(); ov1.fill.fore_color.rgb = RGBColor(254, 250, 224)
    ov1.line.color.rgb = C['dash_yel']; ov1.line.width = Pt(1.0); ov1.line.dash_style = MSO_LINE_DASH_STYLE.DASH

    ov2 = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(2.48), Inches(6.58), Inches(1.06), Inches(0.68))
    ov2.fill.background()
    ov2.line.color.rgb = C['dash_pur']; ov2.line.width = Pt(1.0); ov2.line.dash_style = MSO_LINE_DASH_STYLE.DASH

    add_entity_node(slide, 'company', 2.12, 6.92, r=0.125, label='c_k')
    add_entity_node(slide, 'company', 2.71, 6.98, r=0.125, label='c_i', is_target=True)
    add_entity_node(slide, 'company', 3.28, 6.92, r=0.125, label='c_j')
    add_line_arrow(slide, 2.26, 6.94, 2.57, 6.97, lw=0.85, head=True, tail=True)
    add_line_arrow(slide, 2.85, 6.97, 3.14, 6.94, lw=0.85, head=True, tail=True)
    add_line_arrow(slide, 2.32, 6.58, 2.34, 6.36, lw=0.9, dash=MSO_LINE_DASH_STYLE.DASH, tail=True)
    add_line_arrow(slide, 3.06, 6.58, 3.04, 6.36, lw=0.9, dash=MSO_LINE_DASH_STYLE.DASH, tail=True)

    # -------------------------------------------------------------------------
    # 5. MACRO ARROWS LINKING ECOSYSTEM, ENDOGENOUS, AND EXTERNAL MODULES
    # -------------------------------------------------------------------------
    add_macro_arrow(slide, 3.88, 4.92, 0.34, 0.32, 'right')
    add_macro_arrow(slide, 4.62, 2.55, 0.30, 0.32, 'down')
    add_macro_arrow(slide, 7.56, 2.55, 0.32, 0.32, 'up')

    # -------------------------------------------------------------------------
    # 6. COUPLED MAIN EXTERNAL & PHYSICS CONTAINER (DASHED SUB-DIVIDERS)
    # -------------------------------------------------------------------------
    add_box(slide, 4.26, 2.82, 11.56, 4.58, fill=C['white'], border=C['black'], lw=1.15)
    add_line_arrow(slide, 10.22, 2.84, 10.22, 7.38, color=C['black'], lw=0.95,
                   dash=MSO_LINE_DASH_STYLE.DASH, tail=False)
    add_line_arrow(slide, 4.28, 5.88, 10.22, 5.88, color=C['black'], lw=0.95,
                   dash=MSO_LINE_DASH_STYLE.DASH, tail=False)

    # === 6A. INJECTIVE EQUITY-GOVERNANCE PHYSICS MODULE (IEGP) ===
    r_paths = [
        ('R_1', ('company', 'c_p'), ('company', 'c_k'), ('company', 'c_i'), C['cub_green']),
        ('R_2', ('company', 'c_j'), ('person',  'p_i'), ('company', 'c_i'), C['cub_orange']),
        ('R_3', ('company', 'c_m'), ('sector',  's_i'), ('company', 'c_i'), C['cub_purple']),
        ('R_4', ('company', 'c_m'), ('area',    'a_i'), ('company', 'c_i'), C['cub_blue']),
    ]
    add_box(slide, 6.50, 2.98, 0.62, 2.00, fill=None, border=C['black'],
            lw=0.85, dash=MSO_LINE_DASH_STYLE.DASH, rounded=True)

    for idx, (r_lbl, n1, n2, n3, cub_c) in enumerate(r_paths):
        ry = 3.22 + idx * 0.465
        add_label(slide, 4.34, ry - 0.14, 0.34, 0.28, r_lbl, size=8.6, italic=True, bold=True, word_wrap=False)
        lb = slide.shapes.add_shape(MSO_SHAPE.LEFT_BRACKET, Inches(4.70), Inches(ry - 0.16), Inches(0.08), Inches(0.32))
        lb.line.color.rgb = C['black']; lb.line.width = Pt(0.95); lb.fill.background()
        rb = slide.shapes.add_shape(MSO_SHAPE.RIGHT_BRACKET, Inches(6.22), Inches(ry - 0.16), Inches(0.08), Inches(0.32))
        rb.line.color.rgb = C['black']; rb.line.width = Pt(0.95); rb.fill.background()

        add_entity_node(slide, n1[0], 4.94, ry, r=0.125, label=n1[1])
        add_entity_node(slide, n2[0], 5.48, ry, r=0.125, label=n2[1])
        add_entity_node(slide, n3[0], 6.02, ry, r=0.125, label=n3[1], is_target=True)
        add_line_arrow(slide, 5.08, ry, 5.34, ry, lw=1.6 if idx == 0 else 0.9, tail=True)
        add_line_arrow(slide, 5.62, ry, 5.88, ry, lw=1.6 if idx == 0 else 0.9, tail=True)
        add_line_arrow(slide, 6.32, ry, 6.54, ry, lw=0.9, tail=True)

        add_cuboid(slide, 6.68, ry - 0.20, w=0.20, h=0.44, color=cub_c, angle=-40)
        add_line_arrow(slide, 7.12, ry, 7.56, ry, lw=0.85, tail=True)

    inj_bar = add_box(slide, 7.58, 3.12, 0.44, 1.72, fill=C['box_lav'], border=C['black'], lw=0.9)
    set_text(inj_bar, '1-WL Injective\nMoment Pool', size=7.5)

    add_operator_circle(slide, 8.62, 3.98, r=0.105, symbol='⊗', size=9.8)
    add_line_arrow(slide, 8.02, 3.98, 8.51, 3.98, lw=0.95, tail=True)
    add_line_arrow(slide, 6.81, 2.98, 8.62, 2.98, lw=0.85, tail=False)
    add_line_arrow(slide, 8.62, 2.98, 8.62, 3.86, lw=0.85, tail=True)

    # Bottom 1-WL Multiset Physics Database Loop
    add_line_arrow(slide, 4.75, 4.90, 4.75, 5.35, lw=0.85, tail=False)
    add_line_arrow(slide, 4.75, 5.35, 6.45, 5.35, lw=0.85, tail=True)
    add_label(slide, 4.82, 5.00, 1.55, 0.34, 'Multiset Scale &\nDegree Extraction', size=7.6)

    add_db_cylinders_icon(slide, 6.55, 5.08, scale=0.96)
    add_label(slide, 6.12, 5.40, 1.38, 0.22, 'HHI & Capital Bank', size=7.7, bold=True)

    add_line_arrow(slide, 7.02, 5.35, 8.62, 5.35, lw=0.85, tail=False)
    add_line_arrow(slide, 8.62, 5.35, 8.62, 4.10, lw=0.85, tail=True)
    add_label(slide, 7.18, 5.00, 1.38, 0.34, 'Injective Scale\nModulation', size=7.6)

    add_line_arrow(slide, 8.73, 3.98, 9.18, 3.98, lw=1.05, tail=True)
    add_cuboid(slide, 9.38, 3.60, w=0.25, h=0.68, color=C['cub_orange'], angle=-36)
    add_label(slide, 9.04, 4.38, 1.06, 0.42, 'Injective Governance\nEmbedding', size=8.0, bold=True)

    add_label(slide, 6.50, 5.58, 3.65, 0.26,
              'Injective Equity-Governance Physics Module (IEGP)',
              size=9.2, bold=True, italic=True)

    # === 6B. CROSS-FITTED MULTI-KERNEL RKHS MODULE (WITH RICH STRUCTURAL ICONS!) ===
    # Visual 5x5 Cross-Fitting Fold Grid Icon + Box
    add_box(slide, 4.42, 6.00, 1.98, 0.96, fill=C['panel_gray'], border=C['black'], lw=0.85, rounded=True)
    add_kfold_matrix_icon(slide, 4.54, 6.10, cell=0.082, gap=0.014)
    add_label(slide, 5.08, 6.12, 1.28, 0.72, '5-Fold OOF\nCross-Fitting\n(Zero Leakage)', size=7.6, bold=True)

    add_line_arrow(slide, 6.40, 6.48, 6.72, 6.48, lw=1.05, tail=True)

    # Visual Partition-Tree + Kernel Landmark Icon + Box
    add_box(slide, 6.74, 6.00, 2.08, 0.96, fill=C['panel_gray'], border=C['black'], lw=0.85, rounded=True)
    add_mercer_tree_kernel_icon(slide, 6.82, 6.18)
    add_label(slide, 7.46, 6.12, 1.32, 0.72, 'Partition-Tree &\nLaplacian Nyström\nMercer Kernels', size=7.5, bold=True)

    add_line_arrow(slide, 8.82, 6.48, 9.20, 6.48, lw=1.05, tail=True)
    add_cuboid(slide, 9.38, 6.06, w=0.25, h=0.66, color=C['cub_teal'], angle=-36)
    add_label(slide, 9.04, 6.78, 1.06, 0.38, 'Mercer RKHS\nEmbedding', size=8.0, bold=True)

    add_label(slide, 4.48, 7.04, 4.50, 0.28,
              'Cross-Fitted Multi-Kernel RKHS Mapping Module',
              size=9.2, bold=True, italic=True)

    # === 6C. TRI-VIEW EMPIRICAL-BAYES HIERARCHICAL HYPERGRAPH MODULE (TV-HHG) ===
    ov_y = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(10.48), Inches(3.02), Inches(1.22), Inches(1.12))
    ov_y.fill.solid(); ov_y.fill.fore_color.rgb = RGBColor(254, 250, 224)
    ov_y.line.color.rgb = C['dash_yel']; ov_y.line.width = Pt(1.0); ov_y.line.dash_style = MSO_LINE_DASH_STYLE.DASH

    ov_p = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(10.50), Inches(3.74), Inches(1.22), Inches(1.12))
    ov_p.fill.background()
    ov_p.line.color.rgb = C['dash_pur']; ov_p.line.width = Pt(1.0); ov_p.line.dash_style = MSO_LINE_DASH_STYLE.DASH

    add_box(slide, 12.18, 3.08, 0.88, 1.68, fill=RGBColor(255, 244, 230),
            border=C['dash_org'], lw=1.0, dash=MSO_LINE_DASH_STYLE.DASH, rounded=True)

    add_entity_node(slide, 'company',   10.80, 3.38, r=0.13,  label='c_k')
    add_entity_node(slide, 'company',   11.12, 3.92, r=0.135, label='c_i', is_target=True)
    add_entity_node(slide, 'company',   10.82, 4.52, r=0.13,  label='c_j')
    add_entity_node(slide, 'syndicate', 12.56, 3.48, r=0.135, label='ε_P')
    add_entity_node(slide, 'env_white', 12.66, 4.38, r=0.135, label='ε_S')

    add_line_arrow(slide, 10.90, 3.50, 11.04, 3.78, lw=0.9, head=True, tail=True)
    add_line_arrow(slide, 10.90, 4.38, 11.04, 4.06, lw=0.9, head=True, tail=True)
    add_line_arrow(slide, 11.28, 3.84, 12.40, 3.50, lw=0.95, tail=True, arrow_type='arrow')
    add_line_arrow(slide, 12.60, 3.64, 12.64, 4.20, lw=0.95, head=True, tail=True)
    add_line_arrow(slide, 12.50, 4.38, 11.28, 4.00, lw=0.95, tail=True)
    add_label(slide, 10.72, 4.02, 0.28, 0.22, '①', size=8.5, bold=True, word_wrap=False)
    add_label(slide, 11.72, 3.48, 0.28, 0.22, '②', size=8.5, bold=True, word_wrap=False)
    add_label(slide, 12.66, 3.82, 0.28, 0.22, '③', size=8.5, bold=True, word_wrap=False)
    add_label(slide, 11.78, 4.18, 0.28, 0.22, '④', size=8.5, bold=True, word_wrap=False)

    # Top Right: 4-Item Propagation & Backoff Legend Box
    add_box(slide, 13.30, 3.04, 2.34, 1.72, fill=C['white'], border=C['black'], lw=0.9)
    add_line_arrow(slide, 13.48, 3.28, 14.24, 3.28, lw=0.95, head=True, tail=True)
    add_label(slide, 13.38, 3.34, 1.05, 0.36, 'Peer-Level\nFusion', size=7.5)

    add_line_arrow(slide, 14.58, 3.28, 15.36, 3.28, lw=0.95, tail=True, arrow_type='arrow')
    add_label(slide, 14.48, 3.34, 1.08, 0.36, 'LOO-EB\nShrinkage', size=7.5)

    add_line_arrow(slide, 13.48, 4.08, 14.24, 4.08, lw=0.95, tail=True)
    add_label(slide, 13.38, 4.14, 1.05, 0.36, 'Zero-Shot\nBackoff', size=7.5)

    add_label(slide, 14.52, 4.12, 1.02, 0.40, 'Backoff\nOperators', size=8.0, bold=True, italic=True)

    d_arr = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(12.68), Inches(4.86), Inches(0.32), Inches(0.36))
    d_arr.fill.solid(); d_arr.fill.fore_color.rgb = C['white']
    d_arr.line.color.rgb = C['black']; d_arr.line.width = Pt(0.95)

    # Stream 1 (City -> Province)
    add_cuboid(slide, 13.06, 5.34, w=0.18, h=0.40, color=C['cub_blue'], angle=-36)
    add_cuboid(slide, 13.06, 5.56, w=0.18, h=0.40, color=C['cub_blue'], angle=-36)
    add_label(slide, 12.72, 5.34, 0.34, 0.22, 'c_i', size=8.2, bold=True, italic=True, color=C['target_red'], word_wrap=False)
    add_label(slide, 12.72, 5.62, 0.34, 0.22, 'c_k', size=8.2, italic=True, word_wrap=False)

    add_box(slide, 13.66, 5.42, 0.92, 0.42, fill=C['box_lav'], border=C['black'], lw=0.85,
            text='LOO-EB\nShrinkage', font_size=7.4, bold=True)
    add_line_arrow(slide, 13.30, 5.52, 13.66, 5.52, lw=0.85, tail=True)
    add_line_arrow(slide, 13.30, 5.74, 13.66, 5.74, lw=0.85, tail=True)
    add_line_arrow(slide, 14.58, 5.63, 15.02, 5.63, lw=0.9, tail=True)
    add_cuboid(slide, 15.14, 5.32, w=0.21, h=0.48, color=C['cub_yellow'], angle=-36)
    add_label(slide, 15.02, 5.82, 0.58, 0.22, 'e_{prov}', size=8.0, bold=True, italic=True, word_wrap=False)

    # Stream 2 (Industry/Qualify -> Macro-Sector/Syndicate)
    add_cuboid(slide, 13.06, 6.22, w=0.18, h=0.40, color=C['cub_blue'], angle=-36)
    add_cuboid(slide, 13.06, 6.44, w=0.18, h=0.40, color=C['cub_blue'], angle=-36)
    add_label(slide, 12.72, 6.22, 0.34, 0.22, 'c_i', size=8.2, bold=True, italic=True, color=C['target_red'], word_wrap=False)
    add_label(slide, 12.72, 6.50, 0.34, 0.22, 'c_j', size=8.2, italic=True, word_wrap=False)

    add_box(slide, 13.66, 6.30, 0.92, 0.42, fill=C['box_lav'], border=C['black'], lw=0.85,
            text='LOO-EB\nShrinkage', font_size=7.4, bold=True)
    add_line_arrow(slide, 13.30, 6.40, 13.66, 6.40, lw=0.85, tail=True)
    add_line_arrow(slide, 13.30, 6.62, 13.66, 6.62, lw=0.85, tail=True)
    add_line_arrow(slide, 14.58, 6.51, 15.02, 6.51, lw=0.9, tail=True)
    add_cuboid(slide, 15.14, 6.20, w=0.21, h=0.48, color=C['cub_rose'], angle=-36)
    add_label(slide, 15.02, 6.70, 0.58, 0.22, 'e_{synd}', size=8.0, bold=True, italic=True, word_wrap=False)

    add_line_arrow(slide, 15.26, 6.04, 15.26, 6.22, lw=0.85, head=True, tail=True)
    add_box(slide, 11.58, 5.86, 0.86, 0.46, fill=C['box_lav'], border=C['black'], lw=0.9,
            text='Peer-Relative\nFusion', font_size=7.5, bold=True)
    add_line_arrow(slide, 12.76, 5.56, 12.10, 5.86, lw=0.9, tail=True)
    add_line_arrow(slide, 12.76, 6.44, 12.10, 6.32, lw=0.9, tail=True)
    add_line_arrow(slide, 15.24, 5.26, 13.16, 5.26, lw=0.8, dash=MSO_LINE_DASH_STYLE.DASH, tail=True)
    add_line_arrow(slide, 15.24, 6.94, 13.16, 6.94, lw=0.8, dash=MSO_LINE_DASH_STYLE.DASH, tail=True)

    add_line_arrow(slide, 11.58, 6.09, 11.12, 6.09, lw=1.05, tail=True)
    add_cuboid(slide, 10.68, 5.72, w=0.25, h=0.68, color=C['cub_indigo'], angle=-36)
    add_label(slide, 10.36, 6.48, 1.06, 0.40, 'Hierarchical\nContext Emb.', size=8.0, bold=True)

    add_label(slide, 10.38, 7.04, 5.30, 0.28,
              'Tri-View Empirical-Bayes Hierarchical Hypergraph Module (TV-HHG)',
              size=9.2, bold=True, italic=True)


# =============================================================================
# SLIDE 2: ZOOMED-IN SUB-MODULE MECHANISM ATLAS & REUSABLE ICON SHEET
# =============================================================================
def build_slide2_mechanism_atlas(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    add_label(slide, 0.20, 0.14, 15.60, 0.42,
              'Sub-Module Mechanism Atlas of OrthoSVM-HHG (100% Native Editable Vector Shapes & Icons)',
              size=13.5, bold=True)

    # --- Panel A: Proposition 1 (Row-Softmax Collapse vs 1-WL Injective Physics) ---
    add_box(slide, 0.20, 0.68, 5.05, 6.68, fill=C['white'], border=C['black'], lw=1.15)
    add_label(slide, 0.30, 0.78, 4.85, 0.36,
              '(a) Proposition 1: Row-Softmax Collapse vs. Injective Equity Physics',
              size=10.0, bold=True)

    add_box(slide, 0.38, 1.24, 4.68, 2.42, fill=C['box_red'], border=C['dark_gray'],
            lw=0.85, dash=MSO_LINE_DASH_STYLE.DASH, rounded=True)
    add_label(slide, 0.48, 1.30, 4.48, 0.28, 'Conventional Row-Normalized Softmax Aggregation (Scale Blind)',
              size=8.7, bold=True, color=C['target_red'])
    add_corporate_shield_icon(slide, 0.58, 1.76, scale=1.15)
    add_label(slide, 0.42, 2.14, 1.25, 0.36, 'Inst. Parent c_p\n(¥ 5,000w, 19.1%)', size=7.6, bold=True)
    add_entity_node(slide, 'person', 0.80, 2.86, r=0.145, label='p_1')
    add_entity_node(slide, 'person', 1.22, 2.86, r=0.145, label='p_2')
    add_entity_node(slide, 'person', 1.64, 2.86, r=0.145, label='p_k')
    add_label(slide, 0.46, 3.08, 1.52, 0.34, 'Scattered Persons\n(¥ 5w each, deg = k)', size=7.6)

    add_line_arrow(slide, 1.62, 1.96, 2.38, 2.34, lw=1.0, tail=True)
    add_line_arrow(slide, 1.88, 2.86, 2.38, 2.52, lw=1.0, tail=True)
    add_box(slide, 2.42, 2.18, 1.18, 0.52, fill=C['white'], border=C['black'], lw=0.9,
            text='Row Softmax\n∑ α_{ji} ≡ 1.00', font_size=8.0, bold=True)
    add_line_arrow(slide, 3.60, 2.44, 4.05, 2.44, lw=1.0, tail=True)
    add_cuboid(slide, 4.22, 2.10, w=0.24, h=0.62, color=C['mid_gray'], angle=-36)
    add_label(slide, 3.90, 2.82, 1.08, 0.48, 'Collapsed Emb.\n(Scale & k Lost!)', size=7.6,
              bold=True, color=C['target_red'])

    add_box(slide, 0.38, 3.92, 4.68, 2.96, fill=C['box_green'], border=C['dark_gray'],
            lw=0.85, rounded=True)
    add_label(slide, 0.48, 3.98, 4.48, 0.28, '1-WL Injective Equity-Governance Physics Encoder (IEGP)',
              size=8.7, bold=True, color=C['area'])
    add_coin_bag_icon(slide, 0.58, 4.48, scale=1.15)
    add_corporate_shield_icon(slide, 0.58, 5.18, scale=1.15)
    add_db_cylinders_icon(slide, 0.55, 5.95, scale=1.05)

    moments = [
        'Cardinality Moment:  log(1 + d_{i,r})',
        'Total & Envelope:  ∑ w_{ji}, max w, min w',
        'Inst. Moat & HHI:  W_{CC} / W_{tot}, ∑ s_j^{2}',
    ]
    for idx, m_txt in enumerate(moments):
        my = 4.44 + idx * 0.72
        add_box(slide, 1.16, my, 2.46, 0.48, fill=C['white'], border=C['black'], lw=0.85,
                text=m_txt, font_size=7.9, bold=True)
        add_line_arrow(slide, 3.62, my + 0.24, 4.06, 5.38, lw=0.95, tail=True)

    add_cuboid(slide, 4.25, 5.02, w=0.26, h=0.72, color=C['cub_orange'], angle=-36)
    add_label(slide, 3.92, 5.84, 1.08, 0.52, '1-WL Injective\nGovernance Emb.', size=7.8, bold=True)
    add_label(slide, 0.36, 6.96, 4.72, 0.30, 'Preserves 15.6× Institutional Capital Shield & Shareholder Dispersion',
              size=8.3, italic=True, bold=True)

    # --- Panel B: Tri-View LOO Empirical-Bayes Backoff (Zero-Shot Unseen Cities) ---
    add_box(slide, 5.45, 0.68, 5.05, 6.68, fill=C['white'], border=C['black'], lw=1.15)
    add_label(slide, 5.55, 0.78, 4.85, 0.36,
              '(b) Tri-View Hierarchical Hypergraph & Zero-Shot Backoff',
              size=10.0, bold=True)

    add_box(slide, 5.70, 1.28, 4.55, 1.22, fill=RGBColor(255, 246, 230), border=C['dash_org'],
            lw=1.0, dash=MSO_LINE_DASH_STYLE.DASH, rounded=True)
    add_label(slide, 5.78, 1.33, 4.40, 0.24, 'Level-2 Macro Hyperedges (Province / Macro-Sector / Syndicate Group)',
              size=8.1, bold=True)
    l2_specs = [
        ('area',      6.50, 2.02, 'Province A_m'),
        ('sector',    7.98, 2.02, 'Macro-Sec S_m'),
        ('syndicate', 9.46, 2.02, 'Syndicate G_m'),
    ]
    for kind, lx, ly, txt in l2_specs:
        add_entity_node(slide, kind, lx, ly - 0.08, r=0.16)
        add_label(slide, lx - 0.62, ly + 0.14, 1.24, 0.24, txt, size=7.8, bold=True)

    # Clean non-crossing arrows between Level-2 box (bottom y=2.50) and Level-1 box (top y=3.02)
    add_line_arrow(slide, 6.50, 2.50, 6.50, 3.00, color=C['target_red'], lw=1.35, tail=True)
    add_line_arrow(slide, 7.98, 2.50, 7.98, 3.00, lw=1.05, head=True, tail=True)
    add_line_arrow(slide, 9.46, 2.50, 9.46, 3.00, lw=1.05, head=True, tail=True)
    add_label(slide, 6.56, 2.62, 1.15, 0.24, 'Backoff!', size=7.5, bold=True, color=C['target_red'])

    add_box(slide, 5.70, 3.02, 4.55, 1.38, fill=C['panel_gray'], border=C['dark_gray'],
            lw=0.9, dash=MSO_LINE_DASH_STYLE.DASH, rounded=True)
    add_label(slide, 5.78, 3.07, 4.40, 0.24, 'Level-1 Fine Hyperedges (153 Cities / 108 Industries / Controllers)',
              size=8.1, bold=True)
    add_box(slide, 5.88, 3.44, 1.24, 0.72, fill=C['box_red'], border=C['target_red'], lw=1.0, rounded=True,
            text='Unseen City e_l\n(|e_l ∩ D_{tr}| = 0)', font_size=7.6, bold=True, text_color=C['target_red'])
    add_box(slide, 7.36, 3.44, 1.24, 0.72, fill=C['white'], border=C['black'], lw=0.85, rounded=True,
            text='Sub-Industry e_i\n(LOO Prior μ̂_i)', font_size=7.6, bold=True)
    add_box(slide, 8.84, 3.44, 1.24, 0.72, fill=C['white'], border=C['black'], lw=0.85, rounded=True,
            text='Controller e_q\n(C-P-C Peers)', font_size=7.6, bold=True)

    add_box(slide, 5.70, 4.74, 4.55, 0.86, fill=C['box_blue'], border=C['black'], lw=0.9, rounded=True,
            text='LOO Empirical-Bayes Smoothing:\nμ̂_{EB}(i, e_l) = (∑_{j≠i} y_j + κ_l · μ̂_{EB}(e_h)) / (|e_l \\ {i}| + κ_l)',
            font_size=8.2, bold=True, font_name='Cambria Math')
    add_line_arrow(slide, 6.50, 4.16, 6.95, 4.74, lw=1.0, tail=True)
    add_line_arrow(slide, 7.98, 4.16, 7.98, 4.74, lw=1.0, tail=True)
    add_line_arrow(slide, 9.46, 4.16, 9.00, 4.74, lw=1.0, tail=True)

    add_entity_node(slide, 'company', 7.12, 6.26, r=0.18, label='c_i', is_target=True)
    add_label(slide, 5.76, 6.02, 1.18, 0.48, 'Long-Tail SME\n(deg ≤ 1: 28.7%)', size=7.8, bold=True)
    add_line_arrow(slide, 7.60, 5.60, 7.26, 6.08, lw=1.1, tail=True)
    add_line_arrow(slide, 7.32, 6.26, 8.38, 6.26, lw=1.1, tail=True)
    add_cuboid(slide, 8.58, 5.90, w=0.26, h=0.70, color=C['cub_indigo'], angle=-36)
    add_label(slide, 8.22, 6.66, 1.24, 0.40, 'Curvature-Aware\nContext Emb.', size=7.8, bold=True)
    add_label(slide, 5.60, 6.96, 4.75, 0.30, 'Solves 70.4% Long-Tail Sparsity & 24.6% Unseen Prefecture Cities',
              size=8.3, italic=True, bold=True)

    # --- Panel C: Theorem 2 (Hilbert Orthogonal Subspace & Vapnik SVM Geometry) ---
    add_box(slide, 10.70, 0.68, 5.10, 6.68, fill=C['white'], border=C['black'], lw=1.15)
    add_label(slide, 10.80, 0.78, 4.90, 0.36,
              '(c) Theorem 2: Hilbert OSP & Max-Margin Calibration',
              size=10.0, bold=True)

    add_box(slide, 10.90, 1.24, 4.70, 2.58, fill=C['box_blue'], border=C['black'], lw=0.9, rounded=True)
    add_label(slide, 10.98, 1.30, 4.54, 0.26, 'Riemannian Hilbert Space (H, M = U^{⊤}U + εI):  H = S_{nr}^⊥ ⊕_M S_{nr}',
              size=8.3, bold=True)
    ox, oy = 11.98, 3.42
    add_line_arrow(slide, ox - 0.30, oy, ox + 3.00, oy, color=C['area'], lw=1.8, tail=True)
    add_label(slide, ox + 1.55, oy + 0.03, 1.88, 0.30, 'Resilience Subspace S_{nr} = Range(B_{nr})',
              size=7.6, bold=True, color=C['area'])
    add_line_arrow(slide, ox, oy, ox, oy - 1.52, color=C['cub_indigo'], lw=1.8, tail=True)
    add_label(slide, 10.95, oy - 1.80, 1.85, 0.28, 'Pure Net Risk f_r^⊥ = Π_⊥ f_r^{raw}',
              size=7.7, bold=True, color=C['cub_indigo'])
    add_line_arrow(slide, ox, oy, ox + 1.98, oy - 1.52, color=C['cub_red'], lw=1.5,
                   dash=MSO_LINE_DASH_STYLE.DASH, tail=True)
    add_label(slide, ox + 1.55, oy - 1.82, 1.98, 0.36, 'Entangled Raw Risk f_r^{raw}\n(|cos_M| = 0.638)',
              size=7.5, bold=True, color=C['target_red'])
    add_line_arrow(slide, ox + 1.98, oy - 1.52, ox, oy - 1.52, color=C['dark_gray'],
                   lw=1.0, dash=MSO_LINE_DASH_STYLE.ROUND_DOT, tail=True)
    add_line_arrow(slide, ox + 1.98, oy - 1.52, ox + 1.98, oy, color=C['dark_gray'],
                   lw=0.9, dash=MSO_LINE_DASH_STYLE.ROUND_DOT, tail=True)
    add_box(slide, ox, oy - 0.17, 0.17, 0.17, fill=None, border=C['black'], lw=0.9)
    add_label(slide, ox + 0.20, oy - 0.36, 1.75, 0.26, '⟨f_r^⊥, f_{nr}⟩_M ≡ 0  (3.37e-4)',
              size=7.7, bold=True, color=C['cub_indigo'])

    add_svm_hyperplane_icon(slide, 10.92, 4.04, w=2.12, h=1.82, show_caption=True)
    add_box(slide, 10.92, 5.94, 2.12, 0.88, fill=C['panel_gray'], border=C['black'], lw=0.85, rounded=True,
            text='Vapnik Primal Hyperplane:\nd(x) = (⟨w_r, f_r^⊥⟩ − ⟨w_{nr}, f_{nr}⟩ + b) / ‖w_{eff}‖_2',
            font_size=7.5, bold=True)

    add_line_arrow(slide, 13.06, 4.95, 13.42, 4.95, lw=1.1, tail=True)

    add_box(slide, 13.44, 4.04, 2.16, 1.82, fill=RGBColor(254, 246, 228), border=C['black'], lw=0.9, rounded=True)
    add_label(slide, 13.50, 4.10, 2.04, 0.34, 'Unsupervised Saerens-EM\nMacro-Regime Calibration',
              size=8.2, bold=True)
    add_box(slide, 13.78, 4.92, 0.36, 0.62, fill=C['cub_blue'], border=C['black'], lw=0.7)
    add_label(slide, 13.60, 5.56, 0.72, 0.24, 'π_{tr}=0.58', size=7.3, bold=True, word_wrap=False)
    add_line_arrow(slide, 14.22, 5.20, 14.68, 5.20, color=C['target_red'], lw=1.1, tail=True)
    add_box(slide, 14.78, 4.70, 0.36, 0.84, fill=C['cub_red'], border=C['black'], lw=0.7)
    add_label(slide, 14.58, 5.56, 0.78, 0.24, 'π̂_{te}=0.62', size=7.3, bold=True,
              color=C['target_red'], word_wrap=False)

    add_box(slide, 13.44, 5.94, 2.16, 0.88, fill=C['box_green'], border=C['black'], lw=0.85, rounded=True,
            text='Zero Target Labels Used!\nRescues C3854 & Catches C3294',
            font_size=7.8, bold=True)

    add_label(slide, 10.80, 6.96, 4.90, 0.30, 'Smooth Linear Operator (Zero 1/‖f_{nr}‖ Singularity) + Pareto Margin',
              size=8.2, italic=True, bold=True)


def main():
    prs = Presentation()
    prs.slide_width = Inches(16.0)
    prs.slide_height = Inches(7.6)

    build_slide1_overall_architecture(prs)
    build_slide2_mechanism_atlas(prs)

    out_paths = [
        r'd:\SDI\figures\OrthoSVM_HHG_Architecture.pptx',
        r'C:\Users\WANG Zhuoran\.gemini\antigravity\brain\a166df68-d445-4f93-84a8-c078f9478915\OrthoSVM_HHG_Architecture.pptx',
    ]
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        prs.save(p)
        print(f"[Saved Editable PPTX] -> {p}")


if __name__ == '__main__':
    main()
