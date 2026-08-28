"""Build reports/final_report.docx from verified project files only.

Primary training evidence: notebooks/02_training.ipynb (AT8) stdout.
EDA figures: local raw parquet using the same plot recipes as notebooks/01_eda.ipynb
(the EDA notebook has no saved plot outputs). AT8 RMSE bar charts use exact
notebook metric values (Prophet omitted from bars because its RMSE ~2100 would
flatten the other bars; Prophet remains in the tables).

Run from repo root:  python reports/_build_submission_report.py
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "final_report.docx"
FIG = Path(__file__).resolve().parent / "_report_figures"
BLACK = RGBColor(0x00, 0x00, 0x00)
FONT = "Times New Roman"


def _rfonts(run, name=FONT):
    run.font.name = name
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rFonts.set(qn(attr), name)


def set_run(run, *, size=12, bold=False, italic=False):
    _rfonts(run)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = BLACK


def configure_styles(doc: Document):
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(12)
    normal.font.color.rgb = BLACK
    normal._element.rPr.rFonts.set(qn("w:ascii"), FONT)
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    pf = normal.paragraph_format
    pf.space_after = Pt(8)
    pf.line_spacing = 1.15
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    heading_sizes = {1: 14, 2: 13, 3: 12}
    for level, size in heading_sizes.items():
        style = doc.styles[f"Heading {level}"]
        style.font.name = FONT
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = BLACK
        style.font.italic = False
        rPr = style.element.get_or_add_rPr()
        rFonts = rPr.find(qn("w:rFonts"))
        if rFonts is None:
            rFonts = OxmlElement("w:rFonts")
            rPr.append(rFonts)
        for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rFonts.set(qn(attr), FONT)
        color = rPr.find(qn("w:color"))
        if color is None:
            color = OxmlElement("w:color")
            rPr.append(color)
        color.set(qn("w:val"), "000000")
        style.paragraph_format.space_before = Pt(16 if level == 1 else 12)
        style.paragraph_format.space_after = Pt(8)
        style.paragraph_format.keep_with_next = True

    try:
        cap = doc.styles["Caption"]
        cap.font.name = FONT
        cap.font.size = Pt(11)
        cap.font.italic = True
        cap.font.color.rgb = BLACK
        cap.font.bold = False
        rPr = cap.element.get_or_add_rPr()
        rFonts = rPr.find(qn("w:rFonts"))
        if rFonts is None:
            rFonts = OxmlElement("w:rFonts")
            rPr.append(rFonts)
        for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rFonts.set(qn(attr), FONT)
    except KeyError:
        pass


def set_update_fields_on_open(doc: Document):
    settings = doc.settings.element
    el = OxmlElement("w:updateFields")
    el.set(qn("w:val"), "true")
    settings.append(el)


def add_page_number(paragraph):
    run = paragraph.add_run()
    set_run(run, size=11)
    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "end")
    run._element.append(fld1)
    run._element.append(instr)
    run._element.append(fld2)


def add_field(paragraph, instruction: str):
    run = paragraph.add_run()
    set_run(run, size=12)
    r = run._element
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    r.append(begin)
    r.append(instr)
    r.append(sep)
    r.append(end)


def heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        set_run(run, size={1: 14, 2: 13, 3: 12}[level], bold=True)
        run.font.color.rgb = BLACK
    return h


def body(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run(text)
    set_run(run, size=12)
    return p


def centered(doc, text, *, size=12, bold=False, italic=False, space_after=8, space_before=0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run(text)
    set_run(run, size=size, bold=bold, italic=italic)
    return p


def bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.line_spacing = 1.15
        if p.runs:
            p.runs[0].text = item
            set_run(p.runs[0], size=12)
        else:
            run = p.add_run(item)
            set_run(run, size=12)


def numbered(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Number")
        p.paragraph_format.space_after = Pt(4)
        if p.runs:
            p.runs[0].text = item
            set_run(p.runs[0], size=12)
        else:
            run = p.add_run(item)
            set_run(run, size=12)


def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.first_line_indent = Cm(0)
    try:
        p.style = doc.styles["Caption"]
    except KeyError:
        pass
    run = p.add_run(text)
    set_run(run, size=11, italic=True)
    return p


def add_picture(doc, name, width_in=6.3):
    path = FIG / name
    if not path.exists():
        body(doc, f"[Figure file not found in provided material: {name}]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(8)
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width_in))


def set_cell_border(cell):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "8")
        el.set(qn("w:color"), "000000")
        tcBorders.append(el)
    tcPr.append(tcBorders)


def shade_cell(cell, hex_color: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    shd.set(qn("w:val"), "clear")
    tcPr.append(shd)


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        set_run(run, size=12, bold=True)
        shade_cell(hdr[i], "E6E6E6")
        set_cell_border(hdr[i])
    for r_i, row in enumerate(rows):
        cells = table.rows[r_i + 1].cells
        for c_i, val in enumerate(row):
            cells[c_i].text = ""
            p = cells[c_i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(val))
            set_run(run, size=12)
            set_cell_border(cells[c_i])
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    return table


def insert_toc_field(doc, instruction: str):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    add_field(p, instruction)
    note = doc.add_paragraph()
    note.paragraph_format.space_after = Pt(8)
    run = note.add_run(
        "Microsoft Word fills this list when fields are updated "
        "(open the file and allow field update, or press Ctrl+A then F9)."
    )
    set_run(run, size=11, italic=True)


def setup_sections(doc: Document):
    for section in doc.sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.54)
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.different_first_page_header_footer = False
        footer = section.footer
        footer.is_linked_to_previous = False
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = fp.add_run("Pearls AQI Predictor  |  Page ")
        set_run(run, size=11)
        add_page_number(fp)


# ---------------------------------------------------------------------------
# Exact AT8 stdout metrics (notebooks/02_training.ipynb FULL COMPARISON)
# ---------------------------------------------------------------------------
AT8_24 = [
    ("Persistence Baseline", "66.730405", "36.111431", "0.745204"),
    ("XGBoost", "74.364967", "50.019281", "0.683567"),
    ("LightGBM", "76.456676", "52.289149", "0.665515"),
    ("Ensemble_top3_tabular", "77.085201", "53.331851", "0.659993"),
    ("RandomForest", "84.914023", "59.164036", "0.587424"),
    ("Ridge", "87.584091", "63.751444", "0.561069"),
    ("GRU", "219.023846", "146.531186", "-1.740176"),
    ("LSTM", "232.032646", "156.863391", "-2.075345"),
    ("Prophet", "2167.324472", "1736.633538", "-267.777676"),
]
AT8_48 = [
    ("Persistence Baseline", "84.796609", "45.775352", "0.588437"),
    ("XGBoost", "97.213262", "69.199877", "0.459083"),
    ("Ensemble_top3_tabular", "98.112376", "70.283343", "0.449031"),
    ("LightGBM", "101.157242", "72.808352", "0.414302"),
    ("Ridge", "101.382767", "73.917026", "0.411688"),
    ("RandomForest", "118.050540", "86.744712", "0.202344"),
    ("LSTM", "234.142256", "152.548810", "-2.132563"),
    ("GRU", "246.401500", "157.766790", "-2.469181"),
    ("Prophet", "2159.443832", "1727.409530", "-265.909192"),
]
AT8_72 = [
    ("Persistence Baseline", "92.456124", "51.518059", "0.510480"),
    ("Ridge", "104.353359", "77.654450", "0.376391"),
    ("Ensemble_top3_tabular", "107.619632", "79.190920", "0.336742"),
    ("XGBoost", "111.990073", "82.949948", "0.281779"),
    ("LightGBM", "112.211034", "81.688058", "0.278942"),
    ("RandomForest", "116.121902", "85.821522", "0.227804"),
    ("LSTM", "235.968967", "160.365461", "-2.182948"),
    ("GRU", "239.031295", "159.196560", "-2.266098"),
    ("Prophet", "2150.783604", "1715.485984", "-263.906508"),
]


def metric_rows(pairs):
    return [[m, rmse, mae, r2] for m, rmse, mae, r2 in pairs]


def build():
    doc = Document()
    configure_styles(doc)
    set_update_fields_on_open(doc)
    setup_sections(doc)

    # ----- Title page -----
    for _ in range(4):
        doc.add_paragraph()
    centered(doc, "10Pearls Data Science Internship", size=14, bold=True, space_after=12)
    centered(doc, "Project Report", size=14, bold=True, space_after=24)
    centered(
        doc,
        "Pearls AQI Predictor:\nA Feature–Training–Inference Pipeline for\nThree-Day-Ahead Air Quality Forecasting in Lahore",
        size=16,
        bold=True,
        space_after=18,
    )
    centered(
        doc,
        "Continuous US EPA AQI computed from PM2.5  |  Horizons 24 / 48 / 72 hours",
        size=12,
        italic=True,
        space_after=28,
    )
    add_table(
        doc,
        ["Item", "Details from project files"],
        [
            ["Project title", "Pearls AQI Predictor"],
            ["Programme", "10Pearls Data Science Internship"],
            ["Target city", "Lahore (default latitude 31.5497, longitude 74.3436)"],
            ["Primary training notebook", "notebooks/02_training.ipynb"],
            ["EDA notebook", "notebooks/01_eda.ipynb"],
            ["Repository (report draft)", "https://github.com/hamzajawad123/aqi_predictor"],
            ["Document date", "28 August 2026"],
            ["Student name", "Not stated in the provided project files"],
            ["University / course / instructor", "Not stated in the provided project files"],
            ["Group members", "Not stated in the provided project files"],
        ],
    )
    caption(doc, "Table 1: Title-page information taken only from provided project files.")
    body(
        doc,
        "Items that are blank in the table above were not invented. The remainder of this "
        "report uses only facts that appear in the project notebooks, source code, README, "
        "and local raw data snapshot."
    )
    doc.add_page_break()

    # ----- Executive summary -----
    heading(doc, "1. Executive Summary", 1)
    body(
        doc,
        "This report documents a three-day-ahead Air Quality Index (AQI) forecast for Lahore. "
        "The README describes the work as a serverless Feature / Training / Inference (FTI) "
        "pipeline for the 10Pearls Data Science Internship. Pollution concentrations come from "
        "the OpenWeather Air Pollution API. Weather covariates come from Open-Meteo. Engineered "
        "features and models are stored in Hopsworks. Inference is FastAPI; the dashboard is Streamlit."
    )
    body(
        doc,
        "The prediction target used throughout the code is a continuous US EPA AQI computed from "
        "PM2.5. OpenWeather’s native main.aqi field is stored as openweather_aqi_category and is "
        "not the regression target. Models are trained mainly on AQI deltas (future AQI minus "
        "current AQI) and scored on reconstructed absolute AQI using RMSE, MAE and R². A candidate "
        "is eligible for the Hopsworks registry only if it beats a persistence baseline "
        "(future AQI equals current AQI) on all three metrics."
    )
    body(
        doc,
        "The primary training evidence in this report is the executed Colab notebook "
        "notebooks/02_training.ipynb. That run read feature group aqi_features version 3, shape "
        "(46100, 52), timestamps 2020-12-05 18:00:00 to 2026-08-08 06:00:00, with 42 input features. "
        "The season-aligned split was train 2020-12-05 to 2024-06-01 (27947 rows), validation "
        "2024-06-01 to 2025-06-01 (8075 rows), and test 2025-06-01 to 2026-08-08 (10078 rows). "
        "Optuna used 15 trials. Persistence was best on every horizon. The notebook printed "
        "“NO model beat persistence on all 3 metrics” for 24h, 48h and 72h, and the registry "
        "summary was registered=False, winner=None for all three horizons."
    )
    body(
        doc,
        "A later production training window is documented separately in src/config.py "
        "(TRAIN_START_DATE default 2025-04-04) and in reports/_build_final_report.py. Those "
        "numbers are not AT8 notebook stdout and are labelled as such. They are not used to "
        "replace the AT8 tables."
    )

    heading(doc, "2. Table of Contents", 1)
    insert_toc_field(doc, r' TOC \o "1-3" \h \z \u ')

    heading(doc, "3. List of Figures", 1)
    insert_toc_field(doc, r' TOC \c "Figure" \h \z \u ')
    body(
        doc,
        "Figure titles used in this document: Figure 1 AQI distribution and boxplot; "
        "Figure 2 pollutant histograms; Figure 3 AQI over time; Figure 4 AQI versus weather; "
        "Figure 5 correlation heatmap; Figure 6 ACF and PACF; Figure 7 seasonal decomposition; "
        "Figure 8 smog season versus normal season; Figure 9 AT8 test RMSE 24h; "
        "Figure 10 AT8 test RMSE 48h; Figure 11 AT8 test RMSE 72h."
    )

    heading(doc, "4. List of Tables", 1)
    insert_toc_field(doc, r' TOC \c "Table" \h \z \u ')
    body(
        doc,
        "Table titles used in this document run from Table 1 (title-page information) through "
        "the dataset, feature, evaluation and comparison tables in later sections."
    )
    doc.add_page_break()

    # ----- Introduction -----
    heading(doc, "5. Introduction", 1)
    body(
        doc,
        "The README states that the project forecasts Lahore AQI three days ahead and is built "
        "as a serverless FTI pipeline. The architecture diagram in the README is:"
    )
    bullets(
        doc,
        [
            "OpenWeather (pollution) and Open-Meteo (weather) feed a feature pipeline.",
            "The feature pipeline writes to the Hopsworks Feature Store.",
            "A training pipeline writes to the Hopsworks Model Registry.",
            "FastAPI performs inference.",
            "A Streamlit dashboard presents forecasts.",
        ],
    )
    body(
        doc,
        "src/config.py sets CITY_NAME default Lahore, LATITUDE 31.5497, LONGITUDE 74.3436, "
        "DATA_START_DATE 2020-11-27 (OpenWeather air-pollution history start), TARGET_HORIZONS "
        "(24, 48, 72), FORECAST_HORIZON_HOURS 72, FEATURE_GROUP_NAME aqi_features, "
        "FEATURE_GROUP_VERSION default 4, MODEL_NAME aqi_forecaster, and SMOG_SEASON_MONTHS "
        "(10, 11, 12, 1)."
    )
    body(
        doc,
        "The same config file documents a regime break on 2025-04-04: mean hourly |AQI change| "
        "collapses from ~46 to ~4.5 and does not recover, with the same break in pm2_5. "
        "TRAIN_START_DATE defaults to that date. The AT8 notebook did not apply that filter; "
        "it trained on the full feature-group v3 table described in Section 12."
    )

    heading(doc, "6. Problem Statement", 1)
    body(
        doc,
        "The project files frame the problem as forecasting a continuous AQI for Lahore at "
        "24, 48 and 72 hours. src/utils/aqi_calculation.py states why OpenWeather’s main.aqi "
        "is not used as the target: it is a coarse 1–5 categorical index, whereas the project "
        "brief asks for a continuous AQI and for RMSE, MAE and R². Those metrics assume a "
        "numeric target with meaningful distances."
    )
    body(
        doc,
        "src/training_pipeline.py and notebooks/02_training.ipynb treat persistence "
        "(“AQI in N hours equals AQI now”) as the registration baseline. Hourly AQI is strongly "
        "autocorrelated, so persistence is a serious baseline. The notebook helper beats_persistence "
        "requires strictly better RMSE, strictly better MAE, and strictly better R² together."
    )

    heading(doc, "7. Project Objectives", 1)
    body(
        doc,
        "The following objectives are stated in the README and in the training/serving code. "
        "No additional objectives were added."
    )
    bullets(
        doc,
        [
            "Forecast Lahore AQI at 24, 48 and 72 hours.",
            "Use one pollution source (OpenWeather) and one weather source (Open-Meteo) for both historical and hourly paths.",
            "Store engineered features in Hopsworks (feature group aqi_features).",
            "Register aqi_forecaster_{24,48,72}h only when a model beats persistence on RMSE, MAE and R².",
            "Serve forecasts through FastAPI GET /predict and present them in Streamlit (app/Home.py).",
            "Automate hourly feature updates and daily retraining with GitHub Actions.",
        ],
    )

    heading(doc, "8. Project Scope", 1)
    bullets(
        doc,
        [
            "City: Lahore at the default coordinates in src/config.py. The pipeline is location-parameterised; the provided files do not report a multi-city training run.",
            "AQI is computed from PM2.5 only using the 2024 US EPA PM2.5 breakpoint table in src/utils/aqi_calculation.py. That file states a full multi-pollutant EPA maximum was not implemented.",
            "OpenWeather’s 4-day pollution forecast is fetched in src/utils/data_fetch.py for possible later comparison; that file states it is not used by the current training pipeline.",
            "notebooks/02_training.ipynb trains Ridge, RandomForest, XGBoost, LightGBM, a top-3 tabular mean ensemble, LSTM, GRU and Prophet, plus a persistence baseline.",
            "src/training_pipeline.py states that /predict serves only tabular or ensemble payloads that consume one feature row. Prophet needs a date series; LSTM/GRU need a 24-hour window.",
            "Production feature-group version in config and GitHub Actions is 4. The AT8 notebook used version 3.",
        ],
    )

    heading(doc, "9. Project Workflow", 1)
    body(
        doc,
        "The sequence below is the sequence supported by the provided files. Steps that are "
        "common in data science but not present in the files are omitted."
    )
    numbered(
        doc,
        [
            "Problem definition: continuous EPA AQI from PM2.5 at 24/48/72 hours for Lahore.",
            "Data collection: OpenWeather pollution history from 2020-11-27; Open-Meteo weather for the same hours; merge on UTC timestamp.",
            "Raw snapshot: data/raw/aqi_raw_merged.parquet written by python -m src.feature_pipeline raw-snapshot.",
            "Validation: src/utils/data_validation.py before insert (required columns, duplicates, nulls, wide physical bounds).",
            "Data understanding and EDA: notebooks/01_eda.ipynb on the raw snapshot (not the engineered feature group).",
            "Feature engineering: src/utils/feature_engineering.py build_feature_set (hourly grid, time encodings, log1p pollutants, lags, rolling windows, change rates, weather interactions, absolute and delta targets).",
            "Feature store: Hopsworks group aqi_features (v3 in AT8; v4 in current config).",
            "Training: notebooks/02_training.ipynb (AT8) and src/training_pipeline.py.",
            "Evaluation: RMSE, MAE, R² on absolute future AQI versus persistence.",
            "Registration gate: beat persistence on all three metrics or register nothing.",
            "Serving: FastAPI and/or Streamlit reading Hopsworks (src/utils/serving.py).",
        ],
    )

    heading(doc, "10. Dataset Description", 1)
    heading(doc, "10.1 Raw merged snapshot (EDA source)", 2)
    body(
        doc,
        "EDA in notebooks/01_eda.ipynb reads the local raw snapshot via src.utils.raw_io.load_raw_snapshot. "
        "The snapshot used to produce the EDA figures and statistics in this report is "
        "data/raw/aqi_raw_merged.parquet. After loading, the figure script added month and "
        "is_smog_season for plots; those two columns are not stored in the 16-column parquet."
    )
    add_table(
        doc,
        ["Item", "Details from local raw snapshot"],
        [
            ["File", "data/raw/aqi_raw_merged.parquet"],
            ["Rows", "46419"],
            ["Stored columns", "16"],
            ["Timestamp range", "2020-11-27 00:00:00 to 2026-08-14 13:00:00"],
            ["Duplicate timestamps", "0"],
            ["Missing values (all stored columns)", "none"],
            ["Target used in this project", "aqi (US EPA AQI from PM2.5)"],
            ["Reference column (not the target)", "openweather_aqi_category"],
        ],
    )
    caption(doc, "Table 2: Raw snapshot used for EDA figures and descriptive statistics.")
    body(
        doc,
        "Stored columns and data types measured on the loaded snapshot (before adding EDA-only "
        "month / is_smog_season) are:"
    )
    add_table(
        doc,
        ["Column", "dtype on snapshot"],
        [
            ["timestamp", "datetime64[ns]"],
            ["openweather_aqi_category", "float64"],
            ["co", "float64"],
            ["no", "float64"],
            ["no2", "float64"],
            ["o3", "float64"],
            ["so2", "float64"],
            ["pm2_5", "float64"],
            ["pm10", "float64"],
            ["nh3", "float64"],
            ["aqi", "int64"],
            ["temperature", "float64"],
            ["humidity", "int64"],
            ["wind_speed", "float64"],
            ["wind_deg", "int64"],
            ["pressure", "float64"],
        ],
    )
    caption(doc, "Table 3: Raw snapshot columns and dtypes.")
    heading(doc, "10.2 Data sources", 2)
    body(
        doc,
        "src/utils/data_fetch.py states the source decision: pollution and AQI from the "
        "OpenWeather Air Pollution API (current, forecast, historical); weather from Open-Meteo "
        "archive and forecast APIs. Historical OpenWeather air pollution on the free tier begins "
        "27 November 2020; DATA_START_DATE is set to that date so both sources share the first day. "
        "Open-Meteo hourly variables used are temperature_2m, relative_humidity_2m, wind_speed_10m, "
        "wind_direction_10m and surface_pressure, renamed to temperature, humidity, wind_speed, "
        "wind_deg and pressure. All timestamps are normalised to UTC before the merge."
    )
    heading(doc, "10.3 Target variable", 2)
    body(
        doc,
        "aqi is computed in src/utils/aqi_calculation.py from PM2.5 using the 2024-revised US EPA "
        "PM2.5 breakpoint table stored in PM25_BREAKPOINTS_2024:"
    )
    add_table(
        doc,
        ["PM2.5 low (µg/m³)", "PM2.5 high (µg/m³)", "AQI low", "AQI high", "Label in source comment"],
        [
            ["0.0", "9.0", "0", "50", "Good"],
            ["9.1", "35.4", "51", "100", "Moderate"],
            ["35.5", "55.4", "101", "150", "Unhealthy for Sensitive Groups"],
            ["55.5", "125.4", "151", "200", "Unhealthy"],
            ["125.5", "225.4", "201", "300", "Very Unhealthy"],
            ["225.5", "325.4", "301", "500", "Hazardous"],
        ],
    )
    caption(doc, "Table 4: PM2.5 breakpoints copied from src/utils/aqi_calculation.py.")
    body(
        doc,
        "The same file states that concentration is truncated (not rounded) to one decimal place "
        "before lookup, matching EPA methodology. Values above 325.4 µg/m³ are extrapolated from "
        "the last band rather than hard-capped at 500. The file also states the simplification: "
        "official EPA AQI is the maximum across six criteria pollutants; this project uses PM2.5 only."
    )
    heading(doc, "10.4 Feature-group table used by AT8", 2)
    body(
        doc,
        "notebooks/02_training.ipynb printed the following after reading Hopsworks:"
    )
    add_table(
        doc,
        ["Item", "AT8 notebook stdout"],
        [
            ["Feature group", "aqi_features v3"],
            ["Shape", "(46100, 52)"],
            ["Range", "2020-12-05 18:00:00 -> 2026-08-08 06:00:00"],
            ["Delta / target columns confirmed", "aqi_delta_24h, aqi_delta_48h, aqi_delta_72h, aqi_target_24h, aqi_target_48h, aqi_target_72h"],
            ["n_features after dropping timestamp, hour, month, openweather_aqi_category and all six target columns", "42"],
            ["Train rows", "27947"],
            ["Validation rows", "8075"],
            ["Test rows", "10078"],
            ["Train window printed", "2020-12-05->2024-06-01"],
            ["Validation window printed", "2024-06-01->2025-06-01"],
            ["Test window printed", "2025-06-01->2026-08-08"],
        ],
    )
    caption(doc, "Table 5: Feature group and split facts from notebooks/02_training.ipynb.")
    body(
        doc,
        "src/config.py states that v3 has delta targets but was built on a gappy frame, so about "
        "13% of lags/targets spanned the wrong number of hours. v4 is described as the same "
        "features on a strict hourly grid. AT8 used v3; current default FEATURE_GROUP_VERSION is 4."
    )

    heading(doc, "11. Data Understanding", 1)
    body(
        doc,
        "On the local raw snapshot used for EDA figures, the aqi series has the following "
        "descriptive statistics (computed by reports/_make_report_figures.py from the same "
        "parquet the EDA notebook reads):"
    )
    add_table(
        doc,
        ["Statistic", "Value"],
        [
            ["aqi mean", "247.7509855877981"],
            ["aqi median", "177.0"],
            ["aqi standard deviation", "194.00734285333604"],
            ["aqi skewness", "1.907488520950293"],
            ["aqi kurtosis", "3.1202724556139034"],
            ["IQR outlier threshold (Q3 + 1.5 IQR)", "479.0"],
            ["Count of aqi values above that threshold", "5837"],
            ["ADF statistic", "-11.764392029744617"],
            ["ADF p-value", "1.1280420946421232e-21"],
        ],
    )
    caption(doc, "Table 6: AQI descriptive statistics and ADF test on the local raw snapshot.")
    body(
        doc,
        "Yearly mean aqi after resampling to year-end (YE-DEC) on the same snapshot:"
    )
    add_table(
        doc,
        ["Year-end timestamp", "Mean aqi"],
        [
            ["2020-12-31", "438.785818"],
            ["2021-12-31", "263.904701"],
            ["2022-12-31", "277.071900"],
            ["2023-12-31", "280.052722"],
            ["2024-12-31", "251.599010"],
            ["2025-12-31", "202.854102"],
            ["2026-12-31", "166.540349"],
        ],
    )
    caption(
        doc,
        "Table 7: Yearly mean AQI on the raw snapshot. 2020 starts on 27 November; 2026 ends on 14 August.",
    )
    body(
        doc,
        "Smog season is defined in src/config.py as months October–January. On the raw snapshot:"
    )
    add_table(
        doc,
        ["is_smog_season", "count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        [
            ["0 (Normal)", "32406.0", "182.315405", "126.145572", "0.0", "112.0", "159.0", "193.0", "1000.0"],
            ["1 (Smog Oct–Jan)", "14013.0", "399.075145", "234.698541", "52.0", "207.0", "310.0", "553.0", "1000.0"],
        ],
    )
    caption(doc, "Table 8: AQI grouped by smog-season flag on the raw snapshot.")
    body(
        doc,
        "Pearson correlation of numeric columns with aqi on the raw snapshot (month column excluded "
        "from the correlation matrix in the figure script):"
    )
    add_table(
        doc,
        ["Variable", "Correlation with aqi"],
        [
            ["aqi", "1.000000"],
            ["pm2_5", "0.986036"],
            ["pm10", "0.958184"],
            ["co", "0.793539"],
            ["no", "0.560187"],
            ["no2", "0.546778"],
            ["is_smog_season", "0.512918"],
            ["pressure", "0.511338"],
            ["openweather_aqi_category", "0.477453"],
            ["so2", "0.367937"],
            ["nh3", "0.361643"],
            ["humidity", "0.284877"],
            ["wind_deg", "0.092419"],
            ["wind_speed", "-0.240057"],
            ["o3", "-0.266326"],
            ["temperature", "-0.545787"],
        ],
    )
    caption(doc, "Table 9: Correlation with aqi on the raw snapshot.")
    body(
        doc,
        "Pollutant skewness on the raw snapshot, sorted descending: no 3.919896, nh3 3.008166, "
        "so2 2.456016, no2 2.265771, co 1.987650, pm2_5 1.411350, pm10 1.290845, o3 1.257247. "
        "src/utils/feature_engineering.py cites right-skew (skew > 1) as the reason for log1p "
        "on those eight pollutant columns."
    )

    heading(doc, "12. Data Preprocessing", 1)
    heading(doc, "12.1 Validation of raw merged data", 2)
    body(
        doc,
        "src/utils/data_validation.py runs before data are written to Hopsworks. Required columns "
        "are timestamp, aqi, pm2_5, pm10, co, no2, o3, so2, nh3, temperature, humidity, wind_speed, "
        "wind_deg, pressure. Duplicate timestamps are dropped (keep first). Rows with nulls in "
        "required columns are dropped. Out-of-range rows are dropped using these bounds:"
    )
    add_table(
        doc,
        ["Column", "Accepted range in VALID_RANGES"],
        [
            ["aqi", "0 to 1000"],
            ["pm2_5", "0 to 2000 µg/m³"],
            ["pm10", "0 to 2000 µg/m³"],
            ["co", "0 to 50000 µg/m³"],
            ["no2", "0 to 2000 µg/m³"],
            ["o3", "0 to 2000 µg/m³"],
            ["so2", "0 to 2000 µg/m³"],
            ["nh3", "0 to 2000 µg/m³"],
            ["temperature", "−30 to 60 °C"],
            ["humidity", "0 to 100 %"],
            ["wind_speed", "0 to 150 km/h"],
            ["wind_deg", "0 to 360 degrees"],
            ["pressure", "850 to 1100 hPa"],
        ],
    )
    caption(doc, "Table 10: Validation ranges from src/utils/data_validation.py.")
    body(
        doc,
        "The file states the bounds are wide on purpose so genuine smog extremes are not treated "
        "as errors. One bad hour drops that row; it does not abort the entire batch unless "
        "raise_on_error is set."
    )
    heading(doc, "12.2 Hourly grid and short-gap interpolation", 2)
    body(
        doc,
        "src/utils/feature_engineering.py to_hourly_grid reindexes onto a strict hourly calendar "
        "before any lag or target shift. The file states the raw history has ~3,700 missing hours "
        "in 415 outages, and that on the gappy frame shift(−24) spanned 24 hours for only 87% of "
        "rows and up to 264 hours for the rest. Outages of at most MAX_INTERPOLATE_HOURS = 6, "
        "end to end, are linearly interpolated. Longer gaps stay NaN and are dropped later. "
        "Integer-typed columns are rounded immediately after the grid step."
    )
    heading(doc, "12.3 Transforms applied in build_feature_set", 2)
    numbered(
        doc,
        [
            "add_time_features: hour, day_of_week, month, is_weekend, hour_sin, hour_cos, month_sin, month_cos.",
            "add_season_flag: is_smog_season for months in SMOG_SEASON_MONTHS (10, 11, 12, 1).",
            "add_log_pollutants: log1p of clipped-at-zero co, no, no2, o3, so2, pm2_5, pm10, nh3 (raw columns replaced in place).",
            "add_lag_features: aqi lags 1, 3, 6, 24, 168 hours.",
            "add_rolling_features: aqi rolling mean, std, min, max over 3, 6, 24 hours.",
            "add_change_rate: aqi_change_rate_1h and aqi_change_rate_24h.",
            "add_weather_interactions: wind_speed × pm2_5 and humidity × pm2_5.",
            "add_targets: aqi_target_{24,48,72}h and aqi_delta_{24,48,72}h.",
            "Training mode: dropna on the full row. Inference mode: dropna only on feature columns so the latest hours (unknown future targets) remain.",
        ],
    )
    heading(doc, "12.4 Scaling (model-specific)", 2)
    body(
        doc,
        "In notebooks/02_training.ipynb, Ridge is a pipeline of StandardScaler plus Ridge. "
        "LSTM and GRU fit StandardScaler on training features and transform train, validation "
        "and test features. RandomForest, XGBoost and LightGBM in that notebook are not wrapped "
        "in StandardScaler. Prophet is fit on timestamp and aqi only."
    )
    heading(doc, "12.5 Train / validation / test split used in AT8", 2)
    body(
        doc,
        "The notebook uses a chronological split that snaps validation and test starts to 1 June. "
        "The executed print was: train 2020-12-05->2024-06-01 | val 2024-06-01->2025-06-01 | "
        "test 2025-06-01->2026-08-08, with row counts 27947 / 8075 / 10078. Optuna used "
        "TimeSeriesSplit(n_splits=5) on the training fold. RANDOM_STATE = 42. N_TRIALS = 15."
    )
    heading(doc, "12.6 Steps not claimed", 2)
    body(
        doc,
        "The AT8 notebook run did not apply TRAIN_START_DATE = 2025-04-04. It did not print a "
        "delta-shrinkage search (that search exists in src/training_pipeline.py / "
        "src/utils/evaluation.py). Duplicate timestamps on the local raw snapshot were 0, so "
        "duplicate removal was not required for that file."
    )

    heading(doc, "13. Exploratory Data Analysis", 1)
    body(
        doc,
        "notebooks/01_eda.ipynb contains the EDA recipes (univariate plots, pollutant histograms, "
        "AQI over time, AQI versus weather, correlation heatmap, ACF/PACF, seasonal decomposition, "
        "ADF, smog-versus-normal boxplot, and a Findings for FE cell). The notebook file in the "
        "repository has no executed plot outputs. The figures below were therefore produced from "
        "the local raw parquet using those same plot recipes. Values in the captions match "
        "reports/_report_figures/eda_stats.txt."
    )

    heading(doc, "13.1 AQI distribution", 2)
    add_picture(doc, "eda_aqi_dist.png", 6.3)
    caption(doc, "Figure 1: Distribution of aqi (histogram with KDE) and aqi boxplot on the raw snapshot.")
    body(
        doc,
        "What the figure shows: aqi is right-skewed (skewness 1.907488520950293). Mean "
        "247.7509855877981 is above median 177.0. The boxplot shows a long upper tail. "
        "5837 observations lie above the IQR fence 479.0. Maximum aqi in the smog/normal "
        "describe table is 1000.0 in both groups."
    )

    heading(doc, "13.2 Pollutant distributions", 2)
    add_picture(doc, "eda_pollutant_hist.png", 6.3)
    caption(doc, "Figure 2: Histograms of pm2_5, pm10, co, no, no2, o3, so2 and nh3 on the raw snapshot.")
    body(
        doc,
        "What the figure shows: all eight pollutant series are right-skewed. The notebook titles "
        "each panel with that column’s skew. Ranked skewness is given in Section 11. "
        "src/utils/feature_engineering.py applies log1p to these columns for that reason."
    )

    heading(doc, "13.3 AQI over time", 2)
    add_picture(doc, "eda_aqi_time.png", 6.3)
    caption(doc, "Figure 3: aqi over time for Lahore on the raw snapshot (2020-11-27 to 2026-08-14).")
    body(
        doc,
        "What the figure shows: repeated winter peaks and a visible later decline in typical level, "
        "consistent with Table 7 yearly means falling from 438.785818 in incomplete 2020 to "
        "166.540349 in incomplete 2026. src/utils/feature_engineering.py cites a multi-year "
        "downward AQI trend as a reason for delta targets."
    )

    heading(doc, "13.4 AQI versus weather", 2)
    add_picture(doc, "eda_aqi_weather.png", 6.3)
    caption(doc, "Figure 4: Scatter of aqi versus temperature, humidity, wind_speed and pressure.")
    body(
        doc,
        "What the figure shows: panel titles include the Pearson r from the same snapshot. "
        "Table 9 gives temperature −0.545787, humidity 0.284877, wind_speed −0.240057, "
        "pressure 0.511338. The negative temperature association and positive pressure "
        "association are visible as slope direction in the clouds of points. The provided "
        "material does not state a causal mechanism beyond the feature-engineering comments "
        "on wind dispersing pollution and humidity affecting secondary particle formation."
    )

    heading(doc, "13.5 Correlation heatmap", 2)
    add_picture(doc, "eda_corr.png", 5.6)
    caption(doc, "Figure 5: Correlation heatmap of numeric raw columns (month excluded).")
    body(
        doc,
        "What the figure shows: aqi is most strongly associated with pm2_5 (0.986036) and "
        "pm10 (0.958184). That is expected because aqi is computed from pm2_5. "
        "openweather_aqi_category correlates with aqi at 0.477453, weaker than pm2_5, which "
        "matches the code comment that the 1–5 field is not the continuous target."
    )

    heading(doc, "13.6 Autocorrelation", 2)
    add_picture(doc, "eda_acf_pacf.png", 6.3)
    caption(doc, "Figure 6: ACF of aqi up to 168 hourly lags and PACF of aqi up to 72 hourly lags.")
    body(
        doc,
        "What the figure shows: AQI is strongly autocorrelated at short lags, with structure "
        "out to daily (24h) and weekly (168h) ranges in the ACF panel. "
        "src/utils/feature_engineering.py states that lags {1, 3, 6, 24, 168} and rolling "
        "windows {3, 6, 24} were chosen from ACF/PACF on raw AQI."
    )

    heading(doc, "13.7 Seasonal decomposition", 2)
    add_picture(doc, "eda_decompose.png", 6.0)
    caption(doc, "Figure 7: Additive seasonal_decompose of hourly aqi with period=24.")
    body(
        doc,
        "What the figure shows: an observed series, a slow-moving trend, a repeating 24-hour "
        "seasonal component, and a residual. The notebook uses model=\"additive\" and period=24. "
        "The figure script interpolated to hourly frequency before decompose, matching the "
        "EDA notebook recipe (asfreq hourly then interpolate)."
    )

    heading(doc, "13.8 Smog season versus normal season", 2)
    add_picture(doc, "eda_smog.png", 5.2)
    caption(doc, "Figure 8: Boxplot of aqi for Normal versus Smog (Oct–Jan) on the raw snapshot.")
    body(
        doc,
        "What the figure shows: smog-season hours have higher aqi. Table 8 mean 399.075145 "
        "(n=14013) versus 182.315405 (n=32406). This flag is both a model feature and the "
        "stratification key named in feature_engineering.py and training_pipeline.py."
    )

    heading(doc, "14. Feature and Variable Analysis", 1)
    body(
        doc,
        "AT8 used 42 input columns obtained by dropping timestamp, hour, month, "
        "openweather_aqi_category, aqi_target_{24,48,72}h and aqi_delta_{24,48,72}h from the "
        "52-column feature-group table. Reconstructing that 42-column list from "
        "src/utils/feature_engineering.py plus the raw columns stored in the group gives:"
    )
    bullets(
        doc,
        [
            "Pollutants after log1p (replaced in place): co, no, no2, o3, so2, pm2_5, pm10, nh3.",
            "Current aqi (untransformed).",
            "Weather: temperature, humidity, wind_speed, wind_deg, pressure.",
            "Time encodings kept as features: day_of_week, is_weekend, hour_sin, hour_cos, month_sin, month_cos, is_smog_season.",
            "AQI lags: aqi_lag_1h, aqi_lag_3h, aqi_lag_6h, aqi_lag_24h, aqi_lag_168h.",
            "Rolling aqi: mean/std/min/max for windows 3h, 6h and 24h (12 columns).",
            "Change rates: aqi_change_rate_1h, aqi_change_rate_24h.",
            "Interactions: wind_pollutant_interaction, humidity_pollutant_interaction.",
        ],
    )
    body(
        doc,
        "That list has 42 names. The AT8 notebook printed n_features: 42 but did not print the "
        "42 names as a separate list. The names above come from the feature-engineering code "
        "that produces the feature group, together with the drop list in the notebook."
    )
    body(
        doc,
        "Supervised targets in AT8: tabular, LSTM and GRU train on aqi_delta_{h}h. Absolute "
        "metrics use aqi_target_{h}h. Persistence predicts the current aqi column. Prophet is "
        "fit on columns timestamp and aqi (renamed ds, y) and predicts yhat at "
        "test timestamp + h hours."
    )

    heading(doc, "15. Methodology", 1)
    body(
        doc,
        "All AT8 families except Prophet train to predict the AQI change over the horizon. "
        "Absolute prediction is current aqi plus predicted delta (notebook function "
        "reconstruct_absolute). Metrics are sklearn mean_squared_error (then square root for RMSE), "
        "mean_absolute_error, and r2_score on aqi_target_{h}h. Persistence is evaluate(target, current aqi)."
    )
    body(
        doc,
        "Registration rule in the notebook (identical in spirit to src/training_pipeline.py): "
        "eligible models must have RMSE < persistence RMSE, MAE < persistence MAE, and R² > "
        "persistence R². Among eligible models, sort by RMSE, then MAE, then −R². If none are "
        "eligible, print that nothing is registered."
    )
    body(
        doc,
        "Hopsworks login printed in AT8: project URL https://eu-west.cloud.hopsworks.ai:443/p/41103, "
        "feature group aqi_features v3. Hardware prints: TensorFlow GPU Detected: "
        "/physical_device:GPU:0; XGBoost CUDA Acceleration Enabled."
    )

    heading(doc, "16. Approaches Considered and Used", 1)
    heading(doc, "16.1 Persistence baseline", 2)
    body(
        doc,
        "Name: Persistence Baseline. Purpose in this project: the no-change forecast "
        "future AQI = current AQI. Inputs: the aqi column on the test fold. Features: none "
        "beyond current aqi. Training: none. Evaluation: RMSE, MAE, R² on aqi_target_{h}h. "
        "Results: Table 11–13. Decision in AT8: this baseline was the best on every horizon "
        "and was therefore the registration winner by the gate (no learned model registered)."
    )
    heading(doc, "16.2 Optuna-tuned tabular regression on deltas", 2)
    body(
        doc,
        "Name: Ridge, RandomForest, XGBoost, LightGBM. Purpose: predict aqi_delta_{h}h from the "
        "42 features, then add the delta to current aqi. Training: Optuna minimise time-series "
        "CV RMSE, 15 trials, TimeSeriesSplit n_splits=5, then refit on all training rows. "
        "Evaluation: absolute AQI metrics on test. Decision in AT8: none beat persistence on "
        "all three metrics."
    )
    heading(doc, "16.3 Mean ensemble of top-3 tabular models", 2)
    body(
        doc,
        "Name: Ensemble_top3_tabular. Purpose: average the reconstructed absolute predictions "
        "of the three tabular models with lowest test RMSE. Members are chosen after test scoring "
        "in the notebook loop (sorted tabular RMSE, take three). Decision in AT8: not eligible "
        "under the persistence gate."
    )
    heading(doc, "16.4 Sequence models (LSTM and GRU)", 2)
    body(
        doc,
        "Name: LSTM, GRU. Purpose: model 24-hour windows of the 42 scaled features to predict "
        "the horizon delta. SEQUENCE_LENGTH = 24. Architecture in the notebook: Input (24, n_feat), "
        "LSTM or GRU 64 return_sequences=True, Dropout 0.2, LSTM or GRU 32, Dropout 0.2, "
        "Dense 16 relu, Dense 1. Adam learning rate 1e-3, loss mse, metrics mae, epochs 100, "
        "batch_size 128, EarlyStopping on val_loss patience 8 restore_best_weights. Decision in "
        "AT8: both had RMSE above 219 and negative R² on every horizon; not registered."
    )
    heading(doc, "16.5 Prophet on absolute AQI", 2)
    body(
        doc,
        "Name: Prophet. Purpose: forecast absolute aqi as a level series. Fit on train timestamp "
        "and aqi. Settings in the notebook: daily_seasonality=True, weekly_seasonality=True, "
        "yearly_seasonality=False. Future dates are test timestamp plus the horizon in hours. "
        "Decision in AT8: RMSE above 2150 on every horizon; not registered."
    )
    heading(doc, "16.6 Production training window (not AT8)", 2)
    body(
        doc,
        "src/config.py and src/training_pipeline.py describe a second approach: exclude rows "
        "before TRAIN_START_DATE (default 2025-04-04) because of the documented OpenWeather "
        "regime break. That run is not the AT8 stdout. Results attributed to "
        "reports/_build_final_report.py appear in Section 23 only."
    )

    heading(doc, "17. Models Used", 1)
    body(
        doc,
        "Each subsection uses AT8 test metrics exactly as printed in the FULL COMPARISON table. "
        "Printed one-line summaries in the notebook used fewer decimals (for example Ridge 24h "
        "RMSE=87.58); the tables below keep the FULL COMPARISON precision."
    )

    heading(doc, "17.1 Persistence Baseline", 2)
    body(doc, "Purpose: no-change forecast used as the registration gate.")
    body(doc, "Input features: current aqi only.")
    body(doc, "Training: none.")
    body(doc, "Evaluation metrics: RMSE, MAE, R² on absolute future AQI.")
    add_table(
        doc,
        ["Horizon", "RMSE", "MAE", "R2"],
        [
            ["24", "66.730405", "36.111431", "0.745204"],
            ["48", "84.796609", "45.775352", "0.588437"],
            ["72", "92.456124", "51.518059", "0.510480"],
        ],
    )
    caption(doc, "Table 11: Persistence Baseline test metrics from notebooks/02_training.ipynb.")
    body(
        doc,
        "Interpretation: on the AT8 test window, predicting no change already explains a large "
        "share of variance at 24h (R² 0.745204) and still more than half at 72h (R² 0.510480). "
        "RMSE rises with horizon. Decision: this is the AT8 selected forecast under the stated gate."
    )

    heading(doc, "17.2 Ridge", 2)
    body(doc, "Purpose: linear model of the AQI delta with L2 penalty, after StandardScaler.")
    body(doc, "Input features: the 42-column AT8 feature list.")
    body(
        doc,
        "Training: Optuna search alpha in [0.01, 100] log-uniform, 15 trials, then "
        "StandardScaler + Ridge(**best_params, random_state=42) fit on training data."
    )
    add_table(
        doc,
        ["Horizon", "RMSE", "MAE", "R2"],
        [
            ["24", "87.584091", "63.751444", "0.561069"],
            ["48", "101.382767", "73.917026", "0.411688"],
            ["72", "104.353359", "77.654450", "0.376391"],
        ],
    )
    caption(doc, "Table 12: Ridge test metrics from notebooks/02_training.ipynb.")
    body(
        doc,
        "Interpretation: Ridge RMSE and MAE are higher than persistence at every horizon, and R² "
        "is lower. Decision: not selected in AT8."
    )

    heading(doc, "17.3 RandomForest", 2)
    body(doc, "Purpose: bagged trees on the AQI delta.")
    body(doc, "Input features: the 42-column AT8 feature list.")
    body(
        doc,
        "Training: Optuna over n_estimators 100–250 step 50, max_depth in {8, 12, 16}, "
        "min_samples_leaf 1–8, with max_samples=0.8, n_jobs=-1, random_state=42, 15 trials."
    )
    add_table(
        doc,
        ["Horizon", "RMSE", "MAE", "R2"],
        [
            ["24", "84.914023", "59.164036", "0.587424"],
            ["48", "118.050540", "86.744712", "0.202344"],
            ["72", "116.121902", "85.821522", "0.227804"],
        ],
    )
    caption(doc, "Table 13: RandomForest test metrics from notebooks/02_training.ipynb.")
    body(
        doc,
        "Interpretation: worse than persistence on RMSE, MAE and R² at every horizon. "
        "Decision: not selected in AT8."
    )

    heading(doc, "17.4 XGBoost", 2)
    body(doc, "Purpose: gradient-boosted trees on the AQI delta.")
    body(doc, "Input features: the 42-column AT8 feature list.")
    body(
        doc,
        "Training: Optuna over learning_rate 0.01–0.3 log-uniform, max_depth 3–9, "
        "n_estimators 100–400 step 50, subsample 0.6–1.0, colsample_bytree 0.6–1.0, "
        "tree_method=hist, device=cuda in this run, random_state=42, 15 trials."
    )
    add_table(
        doc,
        ["Horizon", "RMSE", "MAE", "R2"],
        [
            ["24", "74.364967", "50.019281", "0.683567"],
            ["48", "97.213262", "69.199877", "0.459083"],
            ["72", "111.990073", "82.949948", "0.281779"],
        ],
    )
    caption(doc, "Table 14: XGBoost test metrics from notebooks/02_training.ipynb.")
    body(
        doc,
        "Interpretation: among learned models, XGBoost had the lowest 24h and 48h RMSE in AT8, "
        "but both RMSE and MAE remain above persistence and R² remains below persistence. "
        "Decision: not selected in AT8."
    )

    heading(doc, "17.5 LightGBM", 2)
    body(doc, "Purpose: gradient-boosted trees on the AQI delta.")
    body(doc, "Input features: the 42-column AT8 feature list.")
    body(
        doc,
        "Training: Optuna over learning_rate 0.01–0.3 log-uniform, num_leaves 16–96, "
        "n_estimators 100–400 step 50, subsample 0.6–1.0, colsample_bytree 0.6–1.0, "
        "n_jobs=-1, verbosity=-1, random_state=42, 15 trials."
    )
    add_table(
        doc,
        ["Horizon", "RMSE", "MAE", "R2"],
        [
            ["24", "76.456676", "52.289149", "0.665515"],
            ["48", "101.157242", "72.808352", "0.414302"],
            ["72", "112.211034", "81.688058", "0.278942"],
        ],
    )
    caption(doc, "Table 15: LightGBM test metrics from notebooks/02_training.ipynb.")
    body(
        doc,
        "Interpretation: close to XGBoost at 24h, still behind persistence on all three metrics. "
        "Decision: not selected in AT8."
    )

    heading(doc, "17.6 Ensemble_top3_tabular", 2)
    body(doc, "Purpose: mean of the three best tabular absolute reconstructions by test RMSE.")
    body(doc, "Input features: same 42 features through the member models.")
    body(doc, "Training: no extra fit; average of already-trained tabular predictions.")
    add_table(
        doc,
        ["Horizon", "RMSE", "MAE", "R2"],
        [
            ["24", "77.085201", "53.331851", "0.659993"],
            ["48", "98.112376", "70.283343", "0.449031"],
            ["72", "107.619632", "79.190920", "0.336742"],
        ],
    )
    caption(doc, "Table 16: Ensemble_top3_tabular test metrics from notebooks/02_training.ipynb.")
    body(
        doc,
        "Interpretation: the ensemble did not beat the best single tabular model at 24h or 48h "
        "and did not beat persistence. Decision: not selected in AT8."
    )

    heading(doc, "17.7 LSTM", 2)
    body(doc, "Purpose: recurrent net on 24-hour windows of scaled features, predicting the delta.")
    body(doc, "Input features: sequences of length 24 of the 42 scaled columns.")
    body(doc, "Training: as specified in Section 16.4.")
    add_table(
        doc,
        ["Horizon", "RMSE", "MAE", "R2"],
        [
            ["24", "232.032646", "156.863391", "-2.075345"],
            ["48", "234.142256", "152.548810", "-2.132563"],
            ["72", "235.968967", "160.365461", "-2.182948"],
        ],
    )
    caption(doc, "Table 17: LSTM test metrics from notebooks/02_training.ipynb.")
    body(
        doc,
        "Interpretation: RMSE more than three times persistence at 24h; R² is negative, so the "
        "forecast is worse than predicting the test-set mean. Decision: not selected in AT8."
    )

    heading(doc, "17.8 GRU", 2)
    body(doc, "Purpose: same sequence setup as LSTM with GRU layers.")
    body(doc, "Input features: sequences of length 24 of the 42 scaled columns.")
    add_table(
        doc,
        ["Horizon", "RMSE", "MAE", "R2"],
        [
            ["24", "219.023846", "146.531186", "-1.740176"],
            ["48", "246.401500", "157.766790", "-2.469181"],
            ["72", "239.031295", "159.196560", "-2.266098"],
        ],
    )
    caption(doc, "Table 18: GRU test metrics from notebooks/02_training.ipynb.")
    body(
        doc,
        "Interpretation: slightly better than LSTM at 24h, still far worse than persistence, "
        "with negative R². Decision: not selected in AT8."
    )

    heading(doc, "17.9 Prophet", 2)
    body(doc, "Purpose: additive time-series model of absolute aqi.")
    body(doc, "Input features: train timestamp (ds) and train aqi (y) only.")
    add_table(
        doc,
        ["Horizon", "RMSE", "MAE", "R2"],
        [
            ["24", "2167.324472", "1736.633538", "-267.777676"],
            ["48", "2159.443832", "1727.409530", "-265.909192"],
            ["72", "2150.783604", "1715.485984", "-263.906508"],
        ],
    )
    caption(doc, "Table 19: Prophet test metrics from notebooks/02_training.ipynb.")
    body(
        doc,
        "Interpretation: RMSE above 2150 AQI points on every horizon. yearly_seasonality was "
        "False in the notebook. The provided files do not print a further fitted-parameter dump "
        "for this failure. Decision: not selected in AT8."
    )

    heading(doc, "18. Model Training", 1)
    body(
        doc,
        "AT8 training order per horizon, from the notebook source: compute persistence; train "
        "Prophet; tune Ridge, RandomForest, XGBoost, LightGBM on training deltas; train LSTM "
        "then GRU; build Ensemble_top3_tabular; print a table sorted by RMSE; apply "
        "pick_best_candidate; if none, print that nothing is registered."
    )
    body(
        doc,
        "The notebook also writes reports/model_comparison_{h}h.csv. Those CSV files are not "
        "present as committed artefacts in the current reports folder listing used for this "
        "build; the numbers in this report come from the notebook cell stdout, which is saved "
        "inside notebooks/02_training.ipynb."
    )

    heading(doc, "19. Model Evaluation", 1)
    body(
        doc,
        "Metrics used in AT8: RMSE, MAE, R² only. Precision, recall, F1, ROC-AUC and "
        "classification accuracy were not computed in this notebook. All scores are on "
        "absolute future AQI, not on the delta."
    )
    body(
        doc,
        "The gate function in the notebook is: RMSE smaller than persistence, MAE smaller than "
        "persistence, and R² larger than persistence. The printed outcome for each horizon was "
        "“NO model beat persistence on all 3 metrics … nothing registered.” The registry "
        "summary table was:"
    )
    add_table(
        doc,
        ["horizon", "registered", "winner"],
        [
            ["24", "False", "None"],
            ["48", "False", "None"],
            ["72", "False", "None"],
        ],
    )
    caption(doc, "Table 20: AT8 registry summary printed by notebooks/02_training.ipynb.")

    heading(doc, "20. Results", 1)
    heading(doc, "20.1 Complete AT8 comparison (24 hours)", 2)
    add_table(doc, ["Model", "RMSE", "MAE", "R2"], metric_rows(AT8_24))
    caption(doc, "Table 21: AT8 test comparison for horizon 24h (FULL COMPARISON stdout).")
    add_picture(doc, "at8_rmse_24h.png", 6.3)
    caption(
        doc,
        "Figure 9: Test RMSE by model at 24h using the AT8 values in Table 21. Persistence is the left-most / top bar after sorting by RMSE. Prophet is omitted from this bar chart because RMSE 2167.324472 would flatten the other bars; Prophet remains in Table 21.",
    )
    body(
        doc,
        "What happened: every learned model has higher RMSE and higher MAE than persistence, "
        "and lower R². XGBoost is the closest learned model (RMSE 74.364967 versus 66.730405)."
    )

    heading(doc, "20.2 Complete AT8 comparison (48 hours)", 2)
    add_table(doc, ["Model", "RMSE", "MAE", "R2"], metric_rows(AT8_48))
    caption(doc, "Table 22: AT8 test comparison for horizon 48h (FULL COMPARISON stdout).")
    add_picture(doc, "at8_rmse_48h.png", 6.3)
    caption(
        doc,
        "Figure 10: Test RMSE by model at 48h using the AT8 values in Table 22. Prophet omitted from bars for the same scale reason as Figure 9.",
    )
    body(
        doc,
        "What happened: persistence remains first (RMSE 84.796609). XGBoost is again the closest "
        "learned model (97.213262). LSTM, GRU and Prophet remain far worse."
    )

    heading(doc, "20.3 Complete AT8 comparison (72 hours)", 2)
    add_table(doc, ["Model", "RMSE", "MAE", "R2"], metric_rows(AT8_72))
    caption(doc, "Table 23: AT8 test comparison for horizon 72h (FULL COMPARISON stdout).")
    add_picture(doc, "at8_rmse_72h.png", 6.3)
    caption(
        doc,
        "Figure 11: Test RMSE by model at 72h using the AT8 values in Table 23. Prophet omitted from bars for the same scale reason as Figure 9.",
    )
    body(
        doc,
        "What happened: persistence RMSE 92.456124 is still lowest. Among learned models, Ridge "
        "has the lowest 72h RMSE (104.353359), still above persistence."
    )

    heading(doc, "21. Model Comparison", 1)
    body(
        doc,
        "Table 24 places the persistence row beside the best learned model by RMSE at each "
        "horizon. “Best learned” here means lowest RMSE among non-persistence rows in the AT8 "
        "table. It is not a registration winner."
    )
    add_table(
        doc,
        ["Horizon", "Persistence RMSE", "Best learned model by RMSE", "That model RMSE", "Selected in AT8"],
        [
            ["24", "66.730405", "XGBoost", "74.364967", "No — persistence wins the gate"],
            ["48", "84.796609", "XGBoost", "97.213262", "No — persistence wins the gate"],
            ["72", "92.456124", "Ridge", "104.353359", "No — persistence wins the gate"],
        ],
    )
    caption(doc, "Table 24: AT8 persistence versus lowest-RMSE learned model. Selected column follows the notebook gate.")

    heading(doc, "22. Best Model / Final Approach Selection (AT8 notebook)", 1)
    body(
        doc,
        "The AT8 notebook selected no learned model. Persistence Baseline is the only forecast "
        "that satisfies the documented rule, because it is the baseline itself and every other "
        "row failed beats_persistence. Exact persistence performance is Table 11. The notebook "
        "printed registered False and winner None for horizons 24, 48 and 72."
    )
    body(
        doc,
        "The reason is the rule in the notebook, not a general claim that persistence is always "
        "a better class of model. On this feature-group v3, full-history, June-aligned test "
        "window, no trained candidate had simultaneously lower RMSE, lower MAE and higher R²."
    )

    heading(doc, "23. Production Training Window Documented Outside AT8", 1)
    body(
        doc,
        "This section is not AT8 stdout. src/config.py states that training across the "
        "2025-04-04 break “is what made every model lose to persistence on the 2025-26 test "
        "period.” Default TRAIN_START_DATE is therefore 2025-04-04. "
        "reports/_build_final_report.py records selected models for that window as follows. "
        "These numbers are reproduced here only as they appear in that file."
    )
    add_table(
        doc,
        ["Horizon", "Selected model (report draft)", "RMSE", "MAE", "R²"],
        [
            ["24 h (day 1)", "LightGBM (L1)", "28.88", "22.63", "0.064"],
            ["48 h (day 2)", "Random Forest", "30.54", "24.89", "−0.055"],
            ["72 h (day 3)", "Ensemble (top-3 tabular)", "31.65", "26.07", "−0.098"],
        ],
    )
    caption(
        doc,
        "Table 25: Approach 1 selected models as written in reports/_build_final_report.py. Not from AT8 stdout.",
    )
    add_table(
        doc,
        ["Horizon", "Persistence RMSE (report draft)", "Selected-model RMSE (report draft)"],
        [
            ["24 h", "34.84", "28.88"],
            ["48 h", "40.94", "30.54"],
            ["72 h", "40.30", "31.65"],
        ],
    )
    caption(
        doc,
        "Table 26: Approach 1 persistence versus selected RMSE as written in reports/_build_final_report.py.",
    )
    body(
        doc,
        "The same draft states that Approach 1 is the production choice after mentor review, "
        "with registry names aqi_forecaster_24h, aqi_forecaster_48h and aqi_forecaster_72h. "
        "This report does not replace Table 21–23 with Table 25–26. The two experiments differ "
        "in training start date and, per config.py, in feature-group version (AT8 used v3; "
        "production default is v4)."
    )

    heading(doc, "24. Why Other Models Were Not Selected", 1)
    body(
        doc,
        "The following reasons apply to the AT8 run. The documented reason, where present, is "
        "the persistence gate printed by the notebook."
    )
    heading(doc, "24.1 Ridge", 2)
    body(
        doc,
        "Actual 24/48/72 RMSE: 87.584091 / 101.382767 / 104.353359 versus persistence "
        "66.730405 / 84.796609 / 92.456124. MAE and R² also fail the gate at every horizon. "
        "The notebook therefore did not register Ridge."
    )
    heading(doc, "24.2 RandomForest", 2)
    body(
        doc,
        "Actual 24/48/72 RMSE: 84.914023 / 118.050540 / 116.121902. All three metrics fail versus "
        "persistence at every horizon. Not registered."
    )
    heading(doc, "24.3 XGBoost", 2)
    body(
        doc,
        "Actual 24/48/72 RMSE: 74.364967 / 97.213262 / 111.990073. Closest learned model at 24h "
        "and 48h, but RMSE and MAE remain higher than persistence and R² remains lower "
        "(24h R² 0.683567 versus 0.745204). Not registered."
    )
    heading(doc, "24.4 LightGBM", 2)
    body(
        doc,
        "Actual 24/48/72 RMSE: 76.456676 / 101.157242 / 112.211034. Fails the three-metric gate. "
        "Not registered."
    )
    heading(doc, "24.5 Ensemble_top3_tabular", 2)
    body(
        doc,
        "Actual 24/48/72 RMSE: 77.085201 / 98.112376 / 107.619632. Fails the three-metric gate. "
        "Not registered."
    )
    heading(doc, "24.6 LSTM", 2)
    body(
        doc,
        "Actual 24/48/72 RMSE: 232.032646 / 234.142256 / 235.968967 with negative R². Fails the "
        "gate. Not registered."
    )
    heading(doc, "24.7 GRU", 2)
    body(
        doc,
        "Actual 24/48/72 RMSE: 219.023846 / 246.401500 / 239.031295 with negative R². Fails the "
        "gate. Not registered."
    )
    heading(doc, "24.8 Prophet", 2)
    body(
        doc,
        "Actual 24/48/72 RMSE: 2167.324472 / 2159.443832 / 2150.783604. Fails the gate by a large "
        "margin. Not registered."
    )
    body(
        doc,
        "The provided AT8 material does not give a separate narrative reason for each family "
        "beyond the metric gate and the printed “nothing registered” lines. No additional "
        "reason is assumed."
    )

    heading(doc, "25. Serving, Dashboard and Automation", 1)
    body(
        doc,
        "The README states that Streamlit reads Hopsworks directly and FastAPI is optional. "
        "Main dashboard file: app/Home.py. Optional API: uvicorn api.main:app. Docker: "
        "docker compose up --build. GitHub Actions: feature_pipeline.yml hourly "
        "(python -m src.feature_pipeline); training_pipeline.yml daily 02:00 UTC "
        "(python -m src.training_pipeline). Both support workflow_dispatch. Feature-group "
        "version is pinned to 4 in the workflow files. src/utils/serving.py sets "
        "HAZARDOUS_THRESHOLD = 151."
    )
    body(
        doc,
        "A live public URL is not present in the README. The README documents Streamlit "
        "Community Cloud settings (main file app/Home.py, requirements app/requirements.txt, "
        "Python 3.11) but does not record a deployed address."
    )

    heading(doc, "26. Key Findings", 1)
    bullets(
        doc,
        [
            "The local raw snapshot has 46419 rows and 16 stored columns from 2020-11-27 00:00:00 to 2026-08-14 13:00:00, with 0 duplicate timestamps and no missing values in stored columns.",
            "aqi on that snapshot is right-skewed (skewness 1.907488520950293), mean 247.7509855877981, median 177.0.",
            "pm2_5 correlation with aqi is 0.986036, consistent with aqi being computed from PM2.5.",
            "Smog-season (Oct–Jan) mean aqi is 399.075145 versus 182.315405 in other months.",
            "ADF on raw aqi: statistic −11.764392029744617, p-value 1.1280420946421232e-21.",
            "AT8 trained on aqi_features v3, 46100 rows, 42 features, June-aligned split 27947 / 8075 / 10078.",
            "AT8 persistence test RMSE/MAE/R²: 24h 66.730405 / 36.111431 / 0.745204; 48h 84.796609 / 45.775352 / 0.588437; 72h 92.456124 / 51.518059 / 0.510480.",
            "No AT8 learned model beat persistence on RMSE and MAE and R². Registry summary: registered False for 24, 48 and 72.",
            "Closest learned 24h model in AT8 was XGBoost (RMSE 74.364967).",
            "Prophet RMSE exceeded 2150 on every AT8 horizon.",
            "LSTM and GRU AT8 R² values were negative on every horizon.",
            "src/config.py documents a 2025-04-04 OpenWeather regime break (mean hourly |AQI change| ~46 to ~4.5) and defaults training to that start date.",
            "A separate report draft records Approach 1 selected RMSE 28.88 / 30.54 / 31.65. Those figures are not AT8 stdout.",
        ],
    )

    heading(doc, "27. Limitations", 1)
    body(
        doc,
        "Only limitations that are written in the project files are listed."
    )
    bullets(
        doc,
        [
            "src/utils/aqi_calculation.py: AQI is PM2.5-only, not the official EPA maximum across six criteria pollutants.",
            "src/config.py: feature group v3 (the AT8 table) was built on a gappy frame so about 13% of lags/targets spanned the wrong number of hours.",
            "src/utils/feature_engineering.py: ~3,700 missing hours in 415 outages in the raw history; long gaps are dropped rather than imputed beyond 6 hours.",
            "notebooks/02_training.ipynb: nothing was registered because no model beat persistence on all three metrics.",
            "src/training_pipeline.py: Prophet, LSTM and GRU cannot be served through a one-row /predict payload.",
            "reports/_build_final_report.py: Approach 1 test months are described as a calm warm-season period; that draft also states negative R² at 48h and 72h on Approach 1.",
            "README: a live hosted URL is not recorded.",
            "notebooks/01_eda.ipynb in the repository has no saved cell outputs; EDA figures in this Word file were regenerated from the local parquet using that notebook’s plot recipes.",
            "Classification-experiment notebooks are not present in the current repository, so classification metrics from the earlier report draft are not treated as AT8 evidence and are not copied here.",
        ],
    )

    heading(doc, "28. Conclusion", 1)
    body(
        doc,
        "The project problem is a 24/48/72-hour AQI forecast for Lahore using OpenWeather "
        "pollution, Open-Meteo weather, Hopsworks storage, and FastAPI/Streamlit serving. "
        "The target is continuous US EPA AQI from PM2.5, not OpenWeather’s 1–5 category."
    )
    body(
        doc,
        "EDA on the local raw snapshot shows a right-skewed aqi series, strong correlation with "
        "pm2_5, higher smog-season means, and strong hourly autocorrelation. Feature engineering "
        "applies log1p pollutants, ACF-justified lags, rolling windows, change rates, cyclical "
        "time features, a smog flag, weather interactions, and delta targets on an hourly grid."
    )
    body(
        doc,
        "The AT8 notebook trained Ridge, RandomForest, XGBoost, LightGBM, a top-3 tabular "
        "ensemble, LSTM, GRU and Prophet against persistence on feature group v3 with a "
        "full-history June-aligned split. Persistence had the best RMSE, MAE and R² at every "
        "horizon. No model was registered. That is the AT8 result."
    )
    body(
        doc,
        "Project source files separately document a post-2025-04-04 training window intended "
        "for production, with selected RMSE values 28.88 / 30.54 / 31.65 recorded in "
        "reports/_build_final_report.py. Those values are not substituted for the AT8 tables."
    )

    heading(doc, "29. References and Source Notes", 1)
    body(
        doc,
        "Only sources that appear in the provided project files are listed. No external papers "
        "were added."
    )
    bullets(
        doc,
        [
            "README.md — project title, architecture, commands, layout.",
            "notebooks/02_training.ipynb — AT8 training run, splits, metrics, registry summary.",
            "notebooks/01_eda.ipynb — EDA plot recipes and Findings for FE cell (no saved outputs).",
            "notebooks/README.md — notebook roles.",
            "src/config.py — city, dates, feature-group versions, regime break, smog months.",
            "src/utils/data_fetch.py — OpenWeather and Open-Meteo source decision.",
            "src/utils/data_validation.py — required columns and valid ranges.",
            "src/utils/aqi_calculation.py — EPA PM2.5 breakpoints and PM2.5-only simplification; comments cite the EPA Federal Register 2024 PM NAAQS reconsideration as the breakpoint source used by the authors of that file.",
            "src/utils/feature_engineering.py — hourly grid, features, targets.",
            "src/training_pipeline.py — production training loop, persistence gate, serveable kinds.",
            "src/utils/evaluation.py — RMSE/MAE/R², reconstruct_absolute, shrinkage helper (used by the pipeline; not printed in AT8 stdout).",
            "src/utils/serving.py — HAZARDOUS_THRESHOLD = 151.",
            "reports/_build_final_report.py — Approach 1 numbers reproduced in Section 23; repository URL on the title page.",
            "data/raw/aqi_raw_merged.parquet — local snapshot used for EDA figures and Table 2–9.",
            "reports/_report_figures/eda_stats.txt — exact EDA numbers copied into this report.",
        ],
    )
    body(
        doc,
        "OpenWeather and Open-Meteo API URLs appear in src/utils/data_fetch.py. Hopsworks host "
        "default eu-west.cloud.hopsworks.ai appears in src/config.py. The AT8 login print "
        "included https://eu-west.cloud.hopsworks.ai:443/p/41103."
    )

    doc.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
