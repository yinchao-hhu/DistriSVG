#!/usr/bin/env python3
"""Build a Chinese PDF guide explaining the manuscript experiments and figures."""

from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "overleaf" / "figs"
OUTPUT_DIR = ROOT / "output" / "pdf"
OUTPUT_PATH = OUTPUT_DIR / "AP-FedAdam_实验与图表解读说明.pdf"

NAVY = colors.HexColor("#14324A")
BLUE = colors.HexColor("#2F6F9F")
LIGHT_BLUE = colors.HexColor("#EAF3F8")
TEAL = colors.HexColor("#2A7F78")
GREEN = colors.HexColor("#E8F3EC")
AMBER = colors.HexColor("#F4A340")
LIGHT_AMBER = colors.HexColor("#FFF4E4")
RED = colors.HexColor("#B84A45")
LIGHT_RED = colors.HexColor("#FBECEB")
TEXT = colors.HexColor("#22272B")
MUTED = colors.HexColor("#5D6870")
GRID = colors.HexColor("#CBD5DB")


def register_fonts() -> tuple[str, str]:
    regular_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
    ]
    bold_candidates = [
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    ]
    regular = next(path for path in regular_candidates if path.exists())
    bold = next(path for path in bold_candidates if path.exists())
    pdfmetrics.registerFont(TTFont("GuideSans", str(regular)))
    pdfmetrics.registerFont(TTFont("GuideSansBold", str(bold)))
    addMapping("GuideSans", 0, 0, "GuideSans")
    addMapping("GuideSans", 1, 0, "GuideSansBold")
    return "GuideSans", "GuideSansBold"


FONT, FONT_BOLD = register_fonts()


styles = getSampleStyleSheet()
TITLE = ParagraphStyle(
    "TitleCN",
    parent=styles["Title"],
    fontName=FONT_BOLD,
    fontSize=25,
    leading=33,
    textColor=NAVY,
    alignment=TA_LEFT,
    spaceAfter=8,
    wordWrap="CJK",
)
SUBTITLE = ParagraphStyle(
    "SubtitleCN",
    fontName=FONT,
    fontSize=12,
    leading=18,
    textColor=MUTED,
    spaceAfter=10,
    wordWrap="CJK",
)
H1 = ParagraphStyle(
    "H1CN",
    fontName=FONT_BOLD,
    fontSize=18,
    leading=24,
    textColor=NAVY,
    spaceBefore=2,
    spaceAfter=9,
    wordWrap="CJK",
)
H2 = ParagraphStyle(
    "H2CN",
    fontName=FONT_BOLD,
    fontSize=13,
    leading=18,
    textColor=BLUE,
    spaceBefore=5,
    spaceAfter=4,
    wordWrap="CJK",
)
BODY = ParagraphStyle(
    "BodyCN",
    fontName=FONT,
    fontSize=10.2,
    leading=16,
    textColor=TEXT,
    spaceAfter=5,
    wordWrap="CJK",
)
SMALL = ParagraphStyle(
    "SmallCN",
    fontName=FONT,
    fontSize=8.5,
    leading=12.5,
    textColor=MUTED,
    wordWrap="CJK",
)
CAPTION = ParagraphStyle(
    "CaptionCN",
    fontName=FONT,
    fontSize=8.2,
    leading=12,
    textColor=MUTED,
    alignment=TA_CENTER,
    spaceBefore=3,
    spaceAfter=7,
    wordWrap="CJK",
)
COVER_KICKER = ParagraphStyle(
    "CoverKicker",
    fontName=FONT_BOLD,
    fontSize=10,
    leading=14,
    textColor=TEAL,
    spaceAfter=7,
    wordWrap="CJK",
)
CENTER_SMALL = ParagraphStyle(
    "CenterSmall",
    parent=SMALL,
    alignment=TA_CENTER,
)
COMPACT_H = ParagraphStyle(
    "CompactH",
    fontName=FONT_BOLD,
    fontSize=9.5,
    leading=12,
    textColor=BLUE,
    spaceAfter=2,
    wordWrap="CJK",
)
COMPACT_BODY = ParagraphStyle(
    "CompactBody",
    fontName=FONT,
    fontSize=8.3,
    leading=11.5,
    textColor=TEXT,
    wordWrap="CJK",
)
TABLE_HEAD = ParagraphStyle(
    "TableHead",
    fontName=FONT_BOLD,
    fontSize=8.5,
    leading=11,
    textColor=colors.white,
    alignment=TA_CENTER,
    wordWrap="CJK",
)
TABLE_CELL = ParagraphStyle(
    "TableCell",
    fontName=FONT,
    fontSize=8.2,
    leading=11.5,
    textColor=TEXT,
    wordWrap="CJK",
)


def p(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(text, style)


def bullet(text: str) -> Paragraph:
    return Paragraph(f"• {text}", BODY)


def callout(title: str, text: str, fill: colors.Color = LIGHT_BLUE) -> Table:
    content = [
        p(f"<b>{title}</b>", H2),
        p(text, BODY),
    ]
    table = Table([[content]], colWidths=[None])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), fill),
                ("BOX", (0, 0), (-1, -1), 0.7, GRID),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def compact_callout(title: str, text: str, fill: colors.Color = LIGHT_BLUE) -> Table:
    content = [p(f"<b>{title}</b>", COMPACT_H), p(text, COMPACT_BODY)]
    table = Table([[content]], colWidths=[None])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), fill),
                ("BOX", (0, 0), (-1, -1), 0.6, GRID),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def figure(path: Path, max_width: float, max_height: float, caption: str) -> list:
    with PILImage.open(path) as im:
        width_px, height_px = im.size
    scale = min(max_width / width_px, max_height / height_px)
    img = Image(str(path), width=width_px * scale, height=height_px * scale)
    img.hAlign = "CENTER"
    return [img, p(caption, CAPTION)]


def compact_table(data: list[list[str]], widths: list[float]) -> Table:
    formatted = []
    for row_idx, row in enumerate(data):
        style = TABLE_HEAD if row_idx == 0 else TABLE_CELL
        formatted.append([p(str(cell), style) for cell in row])
    table = Table(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.45, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FA")]),
            ]
        )
    )
    return table


def page_decor(canvas, doc) -> None:
    canvas.saveState()
    width, height = canvas._pagesize
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 8.5 * mm, "AP-FedAdam 实验与图表解读说明")
    canvas.drawRightString(width - 18 * mm, 8.5 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def make_doc() -> BaseDocTemplate:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    portrait_frame = Frame(
        18 * mm,
        18 * mm,
        A4[0] - 36 * mm,
        A4[1] - 32 * mm,
        id="portrait_frame",
    )
    landscape_size = landscape(A4)
    landscape_frame = Frame(
        16 * mm,
        18 * mm,
        landscape_size[0] - 32 * mm,
        landscape_size[1] - 32 * mm,
        id="landscape_frame",
    )
    doc = BaseDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="AP-FedAdam 实验与图表解读说明",
        author="Chao Yin, Yinqiao Yan",
        subject="AP-FedAdam manuscript experiment and figure interpretation guide",
    )
    doc.addPageTemplates(
        [
            PageTemplate(id="Portrait", pagesize=A4, frames=[portrait_frame], onPage=page_decor),
            PageTemplate(
                id="Landscape",
                pagesize=landscape_size,
                frames=[landscape_frame],
                onPage=page_decor,
            ),
        ]
    )
    return doc


def add_cover(story: list) -> None:
    story.extend(
        [
            Spacer(1, 18 * mm),
            p("论文实验解读与合作者对照手册", COVER_KICKER),
            p("AP-FedAdam 实验与图表解读说明", TITLE),
            p(
                "Adaptive Proximal Federated Adam for federated cell-type-specific spatially variable gene identification",
                SUBTITLE,
            ),
            Spacer(1, 7 * mm),
            callout(
                "这份文档回答什么？",
                "逐项说明当前论文中每个实验为什么要做、每张图展示什么、结果能够支持什么结论，以及哪些地方目前仍不能过度解读。内容依据当前 manuscript.tex、supplement.tex 和现有实验输出整理。",
                LIGHT_BLUE,
            ),
            Spacer(1, 8 * mm),
            p("<b>当前论文的核心证据链</b>", H2),
            bullet("<b>准确性：</b>正确空间基下能否恢复真实 CTSV 空间系数？"),
            bullet("<b>算法机制：</b>自适应优化、联邦聚合、proximal 和 drift 分别有什么作用？"),
            bullet("<b>稳健性：</b>客户端缺失一种细胞类型时是否仍能分析？"),
            bullet("<b>生物学价值：</b>PDAC 中的候选是否跨空间基稳定且符合已知生物学？"),
            bullet("<b>可推广性：</b>算法能否扩展到与肿瘤完全不同的 mouse brain Visium 数据？"),
            Spacer(1, 10 * mm),
            p("版本日期：2026 年 6 月 23 日", SMALL),
            p("用途：论文内部讨论、实验设计核对、合作者沟通", SMALL),
        ]
    )


def add_overview(story: list) -> None:
    story.extend([PageBreak(), p("一、整体实验逻辑", H1)])
    data = [
        ["证据层级", "对应实验/图", "主要问题", "当前回答"],
        ["参数恢复", "Figures 1-3", "算法能否估准真实空间系数？", "正确空间基下恢复准确；错误空间基显著恶化"],
        ["组件贡献", "Figure 4A-C", "哪些算法组件真正有效？", "Adam 预条件和联邦聚合证据最强"],
        ["稀疏机制", "Figure 4D", "Proximal 是否有价值？", "中等 L1 有益；过强惩罚导致过度稀疏"],
        ["Drift", "长轮次文字结果", "Drift correction 是否改善结果？", "当前同质模拟中没有观察到优势"],
        ["缺失稳健性", "Table 1 + Table S1", "缺失 cell type 时是否还能工作？", "单客户端缺失一种 cell type 时仅小幅退化"],
        ["PDAC 生物学", "Figures 5-6；Tables 2-3", "候选是否稳定、可解释？", "跨基重复且富集已知 PDAC 标记"],
        ["跨组织推广", "Figure 7", "能否用于其他组织？", "优化稳定且候选有空间结构，但仍属探索性"],
    ]
    story.append(compact_table(data, [25 * mm, 31 * mm, 50 * mm, 68 * mm]))
    story.extend(
        [
            Spacer(1, 6 * mm),
            callout(
                "一句话概括当前实验故事",
                "AP-FedAdam 在模型设定合理时能够准确恢复 CTSV 系数；其最明确的算法收益来自 Adam 自适应预条件和跨客户端信息聚合，proximal 在合理调参时提供收缩收益；方法在 PDAC 中得到跨空间基、具有生物学结构的候选，并可探索性推广到脑组织。",
                GREEN,
            ),
            Spacer(1, 5 * mm),
            callout(
                "当前最需要补强的证据",
                "Matched centralized CTSV baseline；真正高客户端异质性条件下的 drift 实验；独立的 zero-inflation/dropout 压力测试图；Mouse Brain 的参考型去卷积和正式 Wald/FDR 推断；PDAC pathway enrichment。",
                LIGHT_AMBER,
            ),
        ]
    )


def add_fig2a(story: list) -> None:
    story.extend([PageBreak(), p("二、Figure 1：空间基选择与总体准确性", H1)])
    story.extend(
        figure(
            FIG_DIR / "fig2a_15combo_mae_heatmap.png",
            170 * mm,
            83 * mm,
            "Figure 1：3 种数据生成模式 × 5 种拟合空间基的系数 MAE 热图。数值越低越好。",
        )
    )
    story.extend(
        [
            p("<b>实验目的</b>", H2),
            p(
                "检验 AP-FedAdam 在不同空间模式下能否恢复真实的细胞类型特异空间系数，同时量化空间基函数设定错误带来的损失。",
                BODY,
            ),
            p("<b>主要结果</b>", H2),
            bullet("linear 数据 + linear 基：MAE = 0.109。"),
            bullet("Gaussian 数据 + Gaussian 基：MAE = 0.243-0.247。"),
            bullet("cosine 数据 + cos2 基：MAE = 0.129。"),
            bullet("空间基明显错误时，MAE 通常上升至 0.56-0.82。"),
            callout(
                "这张图想表达的效果",
                "AP-FedAdam 具有准确恢复 CTSV 空间系数的能力，但空间基选择是决定估计质量的关键因素。论文不能只报告 Wald sensitivity，还必须与 MAE、相关性等估计指标联合解释。",
                GREEN,
            ),
        ]
    )


def add_fig2bc(story: list) -> None:
    story.extend([PageBreak(), p("三、Figures 2-3：逐系数恢复与跨细胞类型稳定性", H1)])
    two_figs = Table(
        [
            [
                figure(FIG_DIR / "fig2b_true_vs_learned_scatter.png", 84 * mm, 78 * mm, "")[0],
                figure(FIG_DIR / "fig2c_celltype_mae_bar.png", 84 * mm, 78 * mm, "")[0],
            ]
        ],
        colWidths=[87 * mm, 87 * mm],
    )
    two_figs.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.extend(
        [
            two_figs,
            p("左：Figure 2，真实系数与估计系数。右：Figure 3，不同细胞类型的平均 MAE。", CAPTION),
            p("<b>Figure 2 的目的与结论</b>", H2),
            p(
                "这张散点图用于证明低 MAE 不是汇总统计造成的假象。在 cosine 数据 + cos2 基的良好设定下，MAE = 0.129、Pearson r = 0.990，说明估计系数在方向和大小上总体接近真实值。",
                BODY,
            ),
            p("<b>Figure 3 的目的与结论</b>", H2),
            p(
                "按 cell type 分层检查误差，避免总体结果被某一种容易估计的细胞类型主导。五种 cell type 的 MAE 约为 0.16-0.20，没有明显失效的 cell type。",
                BODY,
            ),
            callout(
                "两张图共同支持的结论",
                "AP-FedAdam 在正确空间基下不仅总体误差低，而且能够逐系数恢复真实信号，并且这种恢复能力在不同细胞类型之间相对稳定。",
                GREEN,
            ),
            callout(
                "不能过度解读",
                "这些结果是在已知真实模型的模拟中获得，不能直接证明真实组织中的 cell-type assignment 或空间基设定一定正确。",
                LIGHT_RED,
            ),
        ]
    )


def add_fig3(story: list) -> None:
    story.extend([NextPageTemplate("Landscape"), PageBreak(), p("四、Figure 4：算法组件消融与稀疏性压力测试", H1)])
    story.extend(
        figure(
            FIG_DIR / "fig3_ablation_combined.png",
            252 * mm,
            112 * mm,
            "Figure 4：A 训练损失；B 聚合更新范数；C 最终 MAE；D 不同 L1 惩罚下的误差与稀疏比例。",
        )
    )
    story.extend(
        [
            Spacer(1, 4 * mm),
            compact_callout(
                "这组实验想表达的效果",
                "A-B 检查收敛和更新稳定性；C 比较算法组件；D 检验 L1 的偏差-稀疏权衡。当前最明确的算法贡献来自 Adam 自适应预条件和跨客户端聚合。Proximal 的价值表现为适度收缩，而不是惩罚越强越好；当前模拟对 drift correction 的压力不足。",
                GREEN,
            ),
        ]
    )


def add_ablation_detail(story: list) -> None:
    story.extend([NextPageTemplate("Portrait"), PageBreak(), p("五、消融实验的精确解读", H1)])
    data = [
        ["方法", "最终 MAE", "该比较回答什么？"],
        ["AP-FedAdam", "0.108", "完整方法"],
        ["No-prox", "0.109", "初始极小 L1 下 proximal 几乎未产生差异"],
        ["No-drift", "0.109", "当前温和异质性下 drift 暂无明显收益"],
        ["FedAvg", "0.522", "普通本地 SGD 明显不如 Adam 自适应预条件"],
        ["Local-only", "0.220", "不聚合客户端信息会损失估计精度"],
    ]
    story.extend(
        [
            compact_table(data, [38 * mm, 28 * mm, 108 * mm]),
            Spacer(1, 6 * mm),
            p("<b>Proximal 压力测试</b>", H2),
            bullet("最佳 λ = 5×10⁻⁶：AP-FedAdam MAE 0.125，No-prox MAE 0.140。"),
            bullet("λ = 5×10⁻⁵ 时约 75.5% 系数为精确零，但 MAE 增至 0.228。"),
            bullet("λ = 10⁻⁴ 时零比例达到真实的 80%，但 MAE 恶化至 0.347。"),
            callout(
                "正确结论",
                "Proximal 是需要调参的稀疏正则化机制。最佳 λ 下的收益主要来自将小系数向零收缩，而不是精确 support recovery。",
                GREEN,
            ),
            p("<b>Drift 长轮次实验</b>", H2),
            p(
                "在 1000 轮、10 个本地步骤、λ = 5×10⁻⁶ 的设置下，AP-FedAdam 与 No-drift 的 MAE 都为 0.13565，几乎完全相同。",
                BODY,
            ),
            callout(
                "必须诚实保留的限制",
                "当前实验没有证明 drift correction 有效。要支持该组件，需要构造更强的客户端异质性，例如不同客户端具有不同空间模式、信号强度、cell-type composition 或 dropout 机制。",
                LIGHT_RED,
            ),
        ]
    )


def add_missing_ct(story: list) -> None:
    story.extend([PageBreak(), p("六、Table 1：缺失细胞类型稳健性实验", H1)])
    data = [
        ["数据模式", "空间基", "完整 MAE", "缺失 CT MAE", "增加"],
        ["linear", "linear", "0.109", "0.140", "+0.031"],
        ["Gaussian", "gau1", "0.243", "0.274", "+0.031"],
        ["Gaussian", "gau2", "0.247", "0.275", "+0.028"],
        ["cosine", "cos2", "0.129", "0.165", "+0.036"],
        ["全部 15 组合", "平均", "-", "-", "+0.033"],
    ]
    story.extend(
        [
            p("<b>实验目的</b>", H2),
            p(
                "模拟多中心研究中某个机构没有观测到某种细胞类型的情况。服务器对每个参数块只聚合实际拥有该 cell type 的客户端更新，即 block-wise aggregation。",
                BODY,
            ),
            compact_table(data, [32 * mm, 32 * mm, 34 * mm, 38 * mm, 28 * mm]),
            Spacer(1, 6 * mm),
            p("<b>主要结果</b>", H2),
            bullet("一个客户端缺失一种细胞类型后，MAE 增加 0.020-0.050，平均仅增加 0.033。"),
            bullet("正确空间基仍然明显优于错误空间基。"),
            bullet("Wald sensitivity 和经验 Type-I error 基本未改变。"),
            callout(
                "这项实验想表达的效果",
                "AP-FedAdam 不要求所有客户端拥有完全一致的 cell-type panel。Block-wise aggregation 对少量、局部的 cell-type 缺失具有稳健性。",
                GREEN,
            ),
            callout(
                "适用范围",
                "当前只覆盖“8 个客户端中 1 个客户端缺失 1 种细胞类型”的温和情形。大比例客户端缺失、多个 cell type 同时缺失或系统性缺失仍未测试。",
                LIGHT_AMBER,
            ),
        ]
    )


def add_pdac_basis(story: list) -> None:
    story.extend([NextPageTemplate("Landscape"), PageBreak(), p("七、Figure 5：PDAC 空间基敏感性与稳定候选", H1)])
    story.extend(
        figure(
            FIG_DIR / "fig4a_pdac_ctsv_basis_consensus.png",
            205 * mm,
            112 * mm,
            "Figure 5：五种空间基下领先 PDAC gene-cell-type pair 的 -log10(BH q-value)。亮色表示统计证据更强。",
        )
    )
    story.extend(
        [
            compact_callout(
                "实验目的",
                "真实组织中并不知道正确空间基。该实验不强行选择唯一 basis，而是把五种 basis 作为敏感性分析，寻找跨 basis 重复出现的候选。",
                LIGHT_BLUE,
            ),
            compact_callout(
                "主要结论",
                "Linear/Gaussian basis 产生较多发现，cosine basis 更保守。共有 1284 个 pair 在至少 3 种 basis 下显著、26 个在至少 4 种 basis 下显著、5 个在全部 5 种 basis 下显著。跨基稳定性可作为候选可信度的一部分。",
                GREEN,
            ),
        ]
    )


def add_pdac_tables(story: list) -> None:
    story.extend([NextPageTemplate("Portrait"), PageBreak(), p("八、Tables 2-3：PDAC 生物学富集与代表候选", H1)])
    enrich = [
        ["Top N", "已知 marker", "比例", "富集 p"],
        ["20", "6/20", "30%", "1.1×10⁻⁷"],
        ["50", "12/50", "24%", "8.3×10⁻¹³"],
        ["100", "20/100", "20%", "9.8×10⁻¹⁹"],
        ["200", "28/200", "14%", "3.1×10⁻²¹"],
    ]
    candidates = [
        ["Gene / cell type", "共识排名", "显著 basis", "解释"],
        ["SLC29A1 / Cancer.clone.B", "10", "4/5", "hENT1，gemcitabine 相关"],
        ["APOE / mDCs", "22", "4/5", "免疫浸润相关"],
        ["PRSS2 / Ductal", "50", "3/5", "胰腺 trypsinogen"],
        ["POSTN / Fibroblasts", "1100", "3/5", "periostin-positive CAF"],
    ]
    story.extend(
        [
            p("<b>系统 marker enrichment 的目的</b>", H2),
            p(
                "避免只挑几个熟悉基因讲故事，而是检验整个排名前部是否系统性富集已知 PDAC、上皮、基质和免疫 marker。",
                BODY,
            ),
            compact_table(enrich, [30 * mm, 42 * mm, 30 * mm, 42 * mm]),
            Spacer(1, 6 * mm),
            callout(
                "富集实验支持的结论",
                "Top 20 中 30% 是 curated marker，而背景比例约 1.2%。这说明 AP-FedAdam 的高排名基因具有明显生物学结构，而不是随机噪声。",
                GREEN,
            ),
            p("<b>代表候选</b>", H2),
            compact_table(candidates, [61 * mm, 25 * mm, 28 * mm, 60 * mm]),
            Spacer(1, 5 * mm),
            callout(
                "解释边界",
                "Marker annotation 是 gene-level 证据，可支持“排名富集已知生物学”，但不能独立证明每一个 gene-cell-type assignment 都正确。POSTN 的统计排名较弱，主要作为具有明确 CAF 背景的基质代表。",
                LIGHT_AMBER,
            ),
        ]
    )


def add_pdac_maps(story: list) -> None:
    story.extend([NextPageTemplate("Landscape"), PageBreak(), p("九、Figure 6：PDAC 组织空间定位", H1)])
    story.extend(
        figure(
            FIG_DIR / "fig4b_pdac_spatial_component_maps.png",
            250 * mm,
            110 * mm,
            "Figure 6：四个代表 gene-cell-type pair 在六张 PDAC 切片上的空间 proxy 分布。",
        )
    )
    story.extend(
        [
            compact_callout(
                "这张图想表达的效果",
                "导管、恶性、免疫和成纤维细胞相关候选占据不同的组织区域，说明 cell-type-specific spatial analysis 能够把统计发现连接到具体组织结构，而不只是得到普通的 spatially variable gene 列表。",
                GREEN,
            ),
            compact_callout(
                "重要限制",
                "图中显示的是 log(1 + count) × estimated cell-type proportion 的描述性 proxy，不是 AP-FedAdam 拟合得到的 beta 空间效应。因此它适合做组织定位和直观解释，不应作为独立的统计验证。",
                LIGHT_RED,
            ),
        ]
    )


def add_mouse(story: list) -> None:
    story.extend([NextPageTemplate("Landscape"), PageBreak(), p("十、Figure 7：Mouse Brain Visium 跨组织推广性", H1)])
    story.extend(
        figure(
            FIG_DIR / "fig5_mouse_brain_ctsv_summary.png",
            242 * mm,
            105 * mm,
            "Figure 7：左侧为三客户端训练损失；右侧为按最大绝对空间系数排序的 Top 20 gene-cell-type proxy pairs。",
        )
    )
    story.extend(
        [
            compact_callout(
                "实验目的",
                "测试 AP-FedAdam 是否能够从 PDAC 推广到结构和细胞组成完全不同的脑组织，并检查跨三个 Visium 切片的联邦优化是否稳定。",
                LIGHT_BLUE,
            ),
            compact_callout(
                "主要结论",
                "损失从 99.92 降至 2.94。Ttr、Ptgds、Enpp2、Nnat、Fabp7、Tac1 等候选具有已知脑区、脑脊液、胶质或神经元相关空间表达特征，说明结果捕捉到了强空间结构。",
                GREEN,
            ),
            compact_callout(
                "必须保留的探索性限定",
                "Cell-type proportions 来自 marker-score proxy，而不是参考型去卷积；本实验没有 Wald/FDR 推断。Ttr、Ptgds 和血红蛋白基因在多个 proxy cell class 中重复出现，也表明解剖区域与 proxy cell type 之间存在混淆。",
                LIGHT_RED,
            ),
        ]
    )


def add_final(story: list) -> None:
    story.extend([NextPageTemplate("Portrait"), PageBreak(), p("十一、给合作者的结论摘要与下一步", H1)])
    strongest = [
        ["证据", "当前强度", "可在论文中怎样表述"],
        ["正确 basis 下的系数恢复", "强", "AP-FedAdam 可准确恢复 CTSV 空间系数"],
        ["Adam 预条件与联邦聚合", "强", "二者是当前消融中最明确的性能来源"],
        ["Proximal 调参收益", "中等偏强", "中等 L1 改善估计，强 L1 造成偏差"],
        ["缺失 cell type 稳健性", "中等", "温和缺失下 block-wise aggregation 仅小幅退化"],
        ["PDAC 生物学解释", "中等偏强", "跨 basis 候选富集已知 PDAC biology"],
        ["Mouse Brain 推广", "探索性", "证明可运行与候选有空间结构，不是正式 CTSV 验证"],
        ["Drift correction", "弱", "当前实验未显示额外收益"],
    ]
    story.extend(
        [
            compact_table(strongest, [48 * mm, 28 * mm, 98 * mm]),
            Spacer(1, 7 * mm),
            p("<b>建议优先补充的实验</b>", H2),
            bullet("<b>Matched centralized baseline：</b>证明联邦方法相对 pooled-data estimator 的精度代价。"),
            bullet("<b>Client heterogeneity stress test：</b>让 drift correction 真正面对不同空间模式、信号强度或 cell composition。"),
            bullet("<b>Zero-inflation/dropout 图：</b>对应 NB working likelihood 的稳健性主张。"),
            bullet("<b>Mouse Brain 正式验证：</b>参考型去卷积 + Wald/FDR。"),
            bullet("<b>PDAC pathway enrichment：</b>把 gene-level marker 富集升级为通路层面的系统解释。"),
            Spacer(1, 6 * mm),
            callout(
                "合作者讨论时最重要的判断",
                "当前文章最强的故事不是“每个算法组件都已经被证明有效”，而是“一个适用于稀疏 CTSV 推断的联邦自适应 proximal 框架，能够准确恢复模拟信号，并在 PDAC 中得到跨 basis、具有生物学结构的候选”。Drift 和正式 centralized comparison 仍是下一阶段需要补强的部分。",
                LIGHT_BLUE,
            ),
        ]
    )


def build() -> Path:
    doc = make_doc()
    story: list = []
    add_cover(story)
    add_overview(story)
    add_fig2a(story)
    add_fig2bc(story)
    add_fig3(story)
    add_ablation_detail(story)
    add_missing_ct(story)
    add_pdac_basis(story)
    add_pdac_tables(story)
    add_pdac_maps(story)
    add_mouse(story)
    add_final(story)
    doc.build(story)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(build())
