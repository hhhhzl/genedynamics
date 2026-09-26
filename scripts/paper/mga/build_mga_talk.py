"""Plain talk deck: each slide is a title plus a paper figure or a paper equation."""

from pathlib import Path

import pymupdf
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "scripts" / "paper" / "mga" / "output" / "ppt_assets"
ENV = ROOT / "scripts" / "paper" / "mga" / "output" / "env_overview_v4"
OUT = ROOT / "scripts" / "paper" / "mga" / "output" / "MGA_ICLR2027_talk.pptx"
PDF = next(ROOT.glob("_ICLR_2027__Manifold*.pdf"))

INK = RGBColor(0x1C, 0x28, 0x34)
MUTED = RGBColor(0x5C, 0x6B, 0x7A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xD7, 0xDE, 0xE2)

ANCHOR = {MSO_ANCHOR.TOP: "t", MSO_ANCHOR.MIDDLE: "ctr", MSO_ANCHOR.BOTTOM: "b"}


def paint(slide):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = WHITE


def textbox(slide, x, y, w, h, lines, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf._txBody.bodyPr.set("anchor", ANCHOR[anchor])
    for i, line in enumerate(lines):
        paragraph = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        paragraph.alignment = line.get("align", align)
        paragraph.space_before = Pt(line.get("before", 0))
        paragraph.space_after = Pt(line.get("after", 0))
        run = paragraph.add_run()
        run.text = line["text"]
        run.font.size = Pt(line.get("size", 18))
        run.font.bold = line.get("bold", False)
        run.font.color.rgb = line.get("color", INK)
        run.font.name = "Calibri"
    return shape


def rule(slide):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.45), Inches(7.15), Inches(12.4), Pt(0.75)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = LINE
    shape.line.fill.background()


def fit(slide, path, box):
    image = Image.open(path)
    aspect = image.width / image.height
    x, y, w, h = box
    if w / h > aspect:
        height = h
        width = height * aspect
    else:
        width = w
        height = width / aspect
    slide.shapes.add_picture(
        str(path),
        Inches(x + (w - width) / 2),
        Inches(y + (h - height) / 2),
        Inches(width),
        Inches(height),
    )


def footer(slide, page, total):
    rule(slide)
    textbox(
        slide, 0.45, 7.2, 8, 0.24,
        [{"text": "MGA   ·   ICLR 2027, under review", "size": 12, "color": MUTED}],
    )
    textbox(
        slide, 9.8, 7.2, 3.05, 0.24,
        [{"text": f"{page}  /  {total}", "size": 12, "color": MUTED, "align": PP_ALIGN.RIGHT}],
        align=PP_ALIGN.RIGHT,
    )


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def crop_equations():
    """Clip each display formula and white out neighboring body text."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    zoom = pymupdf.Matrix(4.5, 4.5)
    # page, clip, and the vertical band of lines that belong to the formula.
    jobs = {
        "eq1": (1, pymupdf.Rect(104, 610, 520, 676), (622, 671)),
        "eq_primitive": (1, pymupdf.Rect(104, 696, 520, 711), None),
        "eq8": (2, pymupdf.Rect(104, 690, 520, 716), (694, 710)),
        "eq12": (3, pymupdf.Rect(104, 438, 520, 466), (442, 458)),
        "eq13": (3, pymupdf.Rect(104, 646, 520, 674), (648, 672)),
        "eq14": (4, pymupdf.Rect(104, 330, 520, 359), None),
        "eq15": (4, pymupdf.Rect(104, 558, 520, 608), (560, 606)),
        "eq16": (4, pymupdf.Rect(104, 696, 520, 730), (699, 727)),
        "eq17": (5, pymupdf.Rect(104, 124, 520, 150), (128, 147)),
        "lem1": (5, pymupdf.Rect(104, 516, 520, 582), (518, 576)),
        "lem2": (5, pymupdf.Rect(104, 614, 520, 674), (616, 670)),
        "eq20": (6, pymupdf.Rect(104, 500, 520, 556), (504, 552)),
        "alg1": (5, pymupdf.Rect(250, 196, 512, 434), None),
        "table1": (6, pymupdf.Rect(104, 232, 518, 472), None),
    }
    for name, (index, clip, keep) in jobs.items():
        doc = pymupdf.open(PDF)
        page = doc[index]
        if keep is not None:
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    x0, y0, x1, y1 = line["bbox"]
                    if not (y1 < keep[0] or y0 > keep[1]):
                        continue
                    if y1 < clip.y0 or y0 > clip.y1 or x1 < clip.x0 or x0 > clip.x1:
                        continue
                    page.add_redact_annot(pymupdf.Rect(x0 - 1, y0 - 1, x1 + 1, y1 + 1), fill=(1, 1, 1))
            page.apply_redactions()
        pix = page.get_pixmap(matrix=zoom, clip=clip, alpha=False)
        pix.save(str(ASSETS / f"{name}.png"))
        doc.close()


def latest_env(stem):
    numbered = []
    for path in ENV.glob(f"{stem}_*.png"):
        suffix = path.stem.split("_")[-1]
        if suffix.isdigit():
            numbered.append((int(suffix), path))
    return max(numbered)[1]


def content(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    paint(slide)
    return slide


def title_slide(prs, total):
    slide = content(prs)
    textbox(
        slide, 0.6, 2.15, 12, 2.2,
        [
            {"text": "Manifold Generative Annealing", "size": 44, "bold": True, "after": 10},
            {"text": "Contact constraint-aware impedance control", "size": 22, "color": MUTED, "after": 2},
            {"text": "with a reinforcement learning prior", "size": 22, "color": MUTED},
        ],
    )
    textbox(
        slide, 0.6, 6.3, 12, 0.4,
        [{"text": "ICLR 2027   ·   under double-blind review", "size": 16, "color": MUTED}],
    )
    notes(slide, "开场。论文题目是 Manifold Generative Annealing，做接触约束下的阻抗控制，先验来自强化学习。")
    return slide


def visual_slide(prs, title, paths, note, page, total, subtitle=None):
    slide = content(prs)
    textbox(slide, 0.45, 0.28, 12.4, 0.55, [{"text": title, "size": 28, "bold": True}])
    top = 0.95
    if subtitle:
        textbox(slide, 0.45, 0.88, 12.4, 0.45, [{"text": subtitle, "size": 16, "color": MUTED}])
        top = 1.35
    box_h = 6.95 - top
    if len(paths) == 1:
        fit(slide, paths[0], (0.4, top, 12.5, box_h))
    else:
        gap = 0.16
        width = (12.5 - gap * (len(paths) - 1)) / len(paths)
        x = 0.4
        for path in paths:
            fit(slide, path, (x, top, width, box_h))
            x += width + gap
    footer(slide, page, total)
    notes(slide, note)
    return slide


def text_slide(prs, title, paragraphs, note, page, total):
    slide = content(prs)
    textbox(slide, 0.55, 0.4, 12.2, 0.7, [{"text": title, "size": 28, "bold": True}])
    lines = []
    for paragraph in paragraphs:
        lines.append({"text": paragraph, "size": 22, "after": 16})
    textbox(slide, 0.55, 1.4, 12.2, 5.4, lines)
    footer(slide, page, total)
    notes(slide, note)
    return slide


def build():
    crop_equations()
    slides = [
        ("title", {}),
        ("visual", {
            "title": "Planning problem  (1)",
            "paths": [ASSETS / "eq1.png"],
            "subtitle": "Optimize the lower-control sequence. A task controller turns each decision into the robot action.",
            "note": "规划问题是优化有限长的下层控制序列，不是直接采样状态轨迹。约束包括等式 h 和不等式 g。",
        }),
        ("visual", {
            "title": "Motion–impedance primitive",
            "paths": [ASSETS / "eq_primitive.png"],
            "subtitle": "Log-stiffness keeps the realized stiffness positive definite.",
            "note": "接触任务里，每一步决策是参考、对数刚度和前馈量，比如期望力。物理刚度由矩阵指数得到，所以正定。",
        }),
        ("visual", {
            "title": "Prior-weighted target  (8)",
            "paths": [ASSETS / "eq8.png"],
            "subtitle": "The learned sequence distribution is reweighted by the softened model cost.",
            "note": "学习到的轨迹分布乘上模型代价的玻尔兹曼因子。这是对原规划问题的先验加权替代，打分只需要滚动代价。",
        }),
        ("visual", {
            "title": "Reverse step  (12)",
            "paths": [ASSETS / "eq12.png"],
            "subtitle": "A clean-sequence estimate drives the transport. Deterministic updates drop the noise.",
            "note": "反向一步由干净序列的估计驱动，可以是扩散、DDIM 或 flow。后面的几何修正改的就是这个估计。",
        }),
        ("visual", {
            "title": "Online loop",
            "paths": [ASSETS / "fig1_pipeline.png"],
            "subtitle": "Proposals, rollout weighting, then geometric shaping and local residual correction.",
            "note": "总流程：离线训练策略，在线做模型滚动和流形修正，再执行第一个运动、刚度、力原语。",
        }),
        ("visual", {
            "title": "Candidate bank  (13)",
            "paths": [ASSETS / "eq13.png"],
            "subtitle": "Gaussian exploration and learned policy rollouts share one sequence space.",
            "note": "候选库把高斯样本和学习策略的滚动放在同一套下层控制坐标里。M 是随机候选数。",
        }),
        ("visual", {
            "title": "Rollout weighting  (14)",
            "paths": [ASSETS / "eq14.png"],
            "subtitle": "Every candidate is scored by the same closed-loop model. No learned score is required.",
            "note": "权重是代价的 softmax。加权平均得到干净序列估计。代价低不等于组合之后还在任务流形上。",
        }),
        ("visual", {
            "title": "Geometric filter  (15)",
            "paths": [ASSETS / "eq15.png"],
            "subtitle": "The filter attenuates the task-normal part of the update and leaves the tangent part.",
            "note": "滤波器压掉任务法向响应，切向不动。干净估计和反向场都用整形后的更新。",
        }),
        ("visual", {
            "title": "Shaped transport  (16)",
            "paths": [ASSETS / "eq16.png"],
            "subtitle": "The shaped clean estimate is what the reverse step actually follows.",
            "note": "把整形后的估计代进反向传输。因为噪声调度的关系，局部残差校正拿到的是干净坐标里的估计。",
        }),
        ("visual", {
            "title": "Local residual correction  (17)",
            "paths": [ASSETS / "eq17.png"],
            "subtitle": "η is the step gain. ε regularizes the residual Jacobian. This is not a hard-feasibility projection.",
            "note": "LRC 用阻尼雅可比修正消掉有限步留下的残差。它不是精确投影，也不保证硬可行。",
        }),
        ("visual", {
            "title": "One replanning cycle",
            "paths": [ASSETS / "alg1.png"],
            "subtitle": "Weight, shape, transport, correct, then execute the first primitive.",
            "note": "算法把前面的公式串起来：初始化，每个反向步组候选、滚动打分、整形、传输、校正，再执行第一个原语并热启动下一步。",
        }),
        ("visual", {
            "title": "Lemma 1. Finite-bank reweighting  (18)",
            "paths": [ASSETS / "lem1.png"],
            "subtitle": "The softmax weights are the unique solution of this cost-plus-entropy problem.",
            "note": "引理 1：有限候选上的 softmax 权重是代价加熵的唯一解。它说的是候选代价的加权，不是加权均值一定可行。",
        }),
        ("visual", {
            "title": "Lemma 2. Regularized geometric refinement  (19)",
            "paths": [ASSETS / "lem2.png"],
            "subtitle": "The filter is the unique minimizer. Normal response is strictly attenuated.",
            "note": "引理 2：这个滤波器是正则化几何投影的唯一解。核方向保留，非零的法向响应被严格衰减。",
        }),
        ("visual", {
            "title": "Theorem 1. Local residual control  (20)",
            "paths": [ASSETS / "eq20.png"],
            "subtitle": "On a fixed smooth branch, LRC contracts the residual component that the Jacobian can remove.",
            "note": "定理 1：在固定光滑分支上，LRC 按小于 1 的因子收缩可消除的残差。消不掉的分量还在。保证是局部的。",
        }),
        ("visual", {
            "title": "Three contact regimes",
            "paths": [latest_env("scanning"), latest_env("peg_insert"), latest_env("push_to_line")],
            "subtitle": "Surface scanning, peg insertion, and humanoid pushing.",
            "note": "三个环境。扫描有刚体、柔顺、混合和未见形状。插孔有标称、位姿偏移和传感偏移。人形有 15 牛力调节、推到线和带偏航的解卡。",
        }),
        ("visual", {
            "title": "Safe success and contact-load tail",
            "paths": [ASSETS / "table1.png"],
            "subtitle": "S is safe-success rate (%). F is the normalized contact-load tail, nCVaR95. Lower F is better.",
            "note": "主表。S 是安全成功率，F 是接触载荷尾部风险，越低越好。扫描的刚体、柔顺、混合上 MGA 是 100%。未见形状是 50%。插孔标称和传感偏移是 90%，位姿偏移是 50%。",
        }),
        ("visual", {
            "title": "Scanning coverage and force",
            "paths": [ASSETS / "fig3_scan.png"],
            "note": "扫描定性结果。混合表面上 MGA 覆盖是 100%。弯曲路径上，有 LRC 时覆盖 96.0%，去掉 LRC 是 50.5%。",
        }),
        ("visual", {
            "title": "Peg insertion contact",
            "paths": [ASSETS / "fig3_peg.png"],
            "note": "插孔过程、扳手利用和计划选择。MGA 的峰值利用是 0.791，低于去掉先验、DIAL 和 ISSA。",
        }),
        ("visual", {
            "title": "Humanoid force, push, and unjamming",
            "paths": [ASSETS / "fig5_humanoid.png"],
            "note": "人形三个任务。力调节峰值 27.2 牛，其他方法 31 到 59.6 牛，目标是 15 牛。解卡图里去掉 LRC 的轨迹停在安全拒绝，不是执行了不安全轨迹。",
        }),
        ("visual", {
            "title": "Hardware: the model-based component",
            "paths": [ASSETS / "fig4_hardware.png"],
            "subtitle": "Dobot Nova 5. Table, foam, and a curved phantom, then pipe alignment and insertion.",
            "note": "真机是 Dobot Nova 5。扫描桌面、泡沫和曲面体模，再做管子对准和插入。硬件评估的是基于模型的部分，完整学习先验还没有部署。",
        }),
        ("text", {
            "title": "Residual correction keeps contact along the path",
            "paragraphs": [
                "Across the 13 scanning suites, terminal progress stays 99.9% with or without local residual correction.",
                "Contact-valid coverage falls from 96.0% to 84.8% when LRC is removed.",
                "Reaching the end of the path does not mean contact stayed valid along it.",
            ],
            "note": "实验讨论第一问。去掉 LRC 后终点进度几乎不变，但沿途接触有效覆盖从 96.0% 降到 84.8%。",
        }),
        ("text", {
            "title": "The prior helps on some shifts and not on others",
            "paragraphs": [
                "Peg insertion under sensing shift: safe success rises from 70% to 90% when the prior is added, while the force tail rises from 0.726 to 0.780.",
                "Hybrid scanning: the tail falls from 0.738 to 0.658, and safe success is higher.",
                "Socket-pose shift does not improve: 50% with the prior, 60% without it. Unseen shapes stay at 50%.",
            ],
            "note": "实验讨论第二问。先验在传感偏移和混合扫描上有帮助。插座位姿偏移和未见形状没有变得更好。",
        }),
        ("text", {
            "title": "Rejection does not build the next whole-body plan",
            "paragraphs": [
                "Unjamming uses the same safety checks. MGA completes 10/10. Removing the prior leaves 8/10. Removing LRC leaves 3/10.",
                "The seven failed trials without LRC stop because no candidate passes revalidation. They are not unsafe executions.",
                "Fixed-stance pushing stays at 100% safe success. LRC lowers the tail from 0.538 to 0.503.",
            ],
            "note": "实验讨论第三问。安全检查能拒绝坏计划，但不能自己产生下一步。解卡上 LRC 和先验都影响能否继续。",
        }),
        ("text", {
            "title": "Conclusion",
            "paragraphs": [
                "MGA combines structured policy proposals, model-based reassessment, and feedback-conditioned geometric correction in one motion–impedance sequence.",
                "Rollout weighting picks the candidate contribution. Tangent shaping adjusts the update. LRC addresses the residual that remains. The gains of the prior and of LRC depend on the task and the shift.",
                "The method needs informative rollouts and a locally reliable response estimate. Guarantees are local. Hardware tests the model-based component. Deploying the full learned prior is future work.",
            ],
            "note": "结论按论文原文。三步都在同一个序列空间里。收益随任务和偏移变化。理论是局部的。真机还不是完整的学习先验。",
        }),
    ]
    total = len(slides)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    prs.core_properties.title = "Manifold Generative Annealing"
    prs.core_properties.subject = "ICLR 2027 talk"
    for index, (kind, spec) in enumerate(slides, start=1):
        if kind == "title":
            title_slide(prs, total)
        elif kind == "visual":
            visual_slide(prs, spec["title"], spec["paths"], spec["note"], index, total, spec.get("subtitle"))
        else:
            text_slide(prs, spec["title"], spec["paragraphs"], spec["note"], index, total)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
