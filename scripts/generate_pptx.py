import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Theme Colors
    BG_COLOR = RGBColor(10, 17, 40)        # #0A1128 Deep Navy
    CARD_BG = RGBColor(18, 28, 51)         # #121C33 Card container
    CARD_BORDER = RGBColor(35, 50, 85)     # #233255
    TEXT_MAIN = RGBColor(248, 250, 252)    # #F8FAFC
    TEXT_MUTED = RGBColor(148, 163, 184)   # #94A3B8
    TEXT_DIM = RGBColor(100, 116, 139)     # #64748B
    CYAN = RGBColor(56, 189, 248)          # #38BDF8
    EMERALD = RGBColor(16, 185, 129)       # #10B981
    ROSE = RGBColor(244, 63, 94)           # #F43F5E
    INDIGO = RGBColor(129, 140, 248)       # #818CF8
    AMBER = RGBColor(245, 158, 11)         # #F59E0B

    def set_slide_background(slide):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = BG_COLOR

    def add_header(slide, category, title, subtitle=None):
        # Category Tag
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = CYAN
        p_cat.font.name = 'Arial'

        # Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.65))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_MAIN
        p_title.font.name = 'Arial'

        # Subtitle
        if subtitle:
            sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.35), Inches(11.7), Inches(0.4))
            tf_sub = sub_box.text_frame
            tf_sub.word_wrap = True
            p_sub = tf_sub.paragraphs[0]
            p_sub.text = subtitle
            p_sub.font.size = Pt(12)
            p_sub.font.color.rgb = TEXT_MUTED
            p_sub.font.name = 'Arial'

        # Footer
        footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.0), Inches(11.7), Inches(0.35))
        tf_foot = footer_box.text_frame
        p_foot = tf_foot.paragraphs[0]
        p_foot.text = "FairFed Risk • MTech Project Defense • Candidate: Syed Asif • Live: fairfed-risk.vercel.app"
        p_foot.font.size = Pt(9)
        p_foot.font.color.rgb = TEXT_DIM
        p_foot.font.name = 'Arial'

    def add_card(slide, left, top, width, height, title=None, border_color=None):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = border_color if border_color else CARD_BORDER
        card.line.width = Pt(1.5 if border_color else 1)

        if title:
            tbox = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), Inches(0.4))
            p = tbox.text_frame.paragraphs[0]
            p.text = title
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = TEXT_MAIN
            p.font.name = 'Arial'
        return card

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    # Decorative top bar
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.0), Inches(1.2), Inches(0.06))
    bar.fill.solid()
    bar.fill.fore_color.rgb = CYAN
    bar.line.fill.background()

    # Subtitle / Department tag
    tag_box = s1.shapes.add_textbox(Inches(0.8), Inches(1.2), Inches(11.7), Inches(0.4))
    p = tag_box.text_frame.paragraphs[0]
    p.text = "MASTER OF TECHNOLOGY (MTECH) • FINAL DISSERTATION DEFENSE"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN
    p.font.name = 'Arial'

    # Project Title
    t_box = s1.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(1.8))
    p = t_box.text_frame.paragraphs[0]
    p.text = "FairFed Risk: Privacy-Preserving Federated Learning for Multi-Disease Early Risk Prediction under Subgroup Fairness Constraints"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN
    p.font.name = 'Arial'

    # Abstract summary callout
    callout = add_card(s1, Inches(0.8), Inches(3.7), Inches(11.7), Inches(1.3), border_color=CYAN)
    c_box = s1.shapes.add_textbox(Inches(1.0), Inches(3.85), Inches(11.3), Inches(1.0))
    tf = c_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Investigating the fundamental trilemma of Predictive Utility, (ε, δ)-Differential Privacy, and Demographic Fairness across 6 non-IID clinical institutions (N=2,088 patients) predicting Type 2 Diabetes, Coronary Heart Disease, and Chronic Kidney Disease."
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_MUTED
    p.font.name = 'Arial'

    # Meta credentials grid
    meta_box = s1.shapes.add_textbox(Inches(0.8), Inches(5.3), Inches(11.7), Inches(1.5))
    tf = meta_box.text_frame
    
    p = tf.paragraphs[0]
    p.text = "Candidate: Syed Asif   |   Degree: Master of Technology (MTech)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN
    
    p2 = tf.add_paragraph()
    p2.text = "Technology Stack: PyTorch, Flower (FL), Opacus (DP-SGD), Fairlearn, Streamlit, HTML5/Chart.js"
    p2.font.size = Pt(11)
    p2.font.color.rgb = TEXT_MUTED
    p2.space_before = Pt(6)

    p3 = tf.add_paragraph()
    p3.text = "Live Web Suite: https://fairfed-risk.vercel.app   |   Code: github.com/SyedAsif7/fairfed_risk"
    p3.font.size = Pt(11)
    p3.font.color.rgb = EMERALD
    p3.space_before = Pt(4)

    # =========================================================================
    # SLIDE 2: THE HEALTHCARE AI TRILEMMA
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "Problem Formulation", "The Fundamental Healthcare AI Trilemma", "Three conflicting forces prevent direct deployment of clinical deep learning")

    # 3 Column Cards
    col_w = Inches(3.7)
    gap = Inches(0.3)
    left_start = Inches(0.8)

    # Card 1: Data Siloing
    add_card(s2, left_start, Inches(1.9), col_w, Inches(4.7), "1. Regulatory Data Silos", border_color=CYAN)
    tb1 = s2.shapes.add_textbox(left_start + Inches(0.2), Inches(2.6), col_w - Inches(0.4), Inches(3.8))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    bullets1 = [
        "Patient records are protected by strict privacy laws (HIPAA, GDPR, Data Protection Acts).",
        "Raw EHR pooling into centralized cloud servers is legally restricted.",
        "Healthcare institutions cannot share patient-level microdata across institutional boundaries.",
        "Clinical Consequence: Fragmentation impedes large-scale model generalization."
    ]
    for i, b in enumerate(bullets1):
        p = tf1.paragraphs[0] if i == 0 else tf1.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(10)

    # Card 2: Non-IID Skew
    add_card(s2, left_start + col_w + gap, Inches(1.9), col_w, Inches(4.7), "2. Institutional Label Skew", border_color=AMBER)
    tb2 = s2.shapes.add_textbox(left_start + col_w + gap + Inches(0.2), Inches(2.6), col_w - Inches(0.4), Inches(3.8))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    bullets2 = [
        "Real hospitals naturally specialize in specific clinical conditions.",
        "Cardiology centers see severe coronary cases; nephrology clinics see renal dialysis patients.",
        "Severe demographic imbalance: male-heavy cardiology clinics vs. female metabolic centers.",
        "Clinical Consequence: Siloed models suffer generalization collapse (AUROC drops to 0.607)."
    ]
    for i, b in enumerate(bullets2):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(10)

    # Card 3: Privacy-Fairness Conflict
    add_card(s2, left_start + (col_w + gap)*2, Inches(1.9), col_w, Inches(4.7), "3. Privacy-Fairness Conflict", border_color=ROSE)
    tb3 = s2.shapes.add_textbox(left_start + (col_w + gap)*2 + Inches(0.2), Inches(2.6), col_w - Inches(0.4), Inches(3.8))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    bullets3 = [
        "Federated model updates still leak patient identities through gradient inversion attacks.",
        "Bounding disclosures via DP-SGD requires gradient clipping (C) and Gaussian noise (σ).",
        "Clipping heavily reduces signal from minority/underrepresented patient samples.",
        "Clinical Consequence: Strict DP exacerbates diagnostic disparities against demographic minorities."
    ]
    for i, b in enumerate(bullets3):
        p = tf3.paragraphs[0] if i == 0 else tf3.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(10)

    # =========================================================================
    # SLIDE 3: RESEARCH OBJECTIVES
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Scope & Goals", "Research Objectives & Key Contributions", "Systematic resolution of the healthcare AI trilemma")

    cards_data = [
        ("Objective 1: Multi-Center Harmonization", "Harmonize 3 distinct clinical cohorts (UCI Heart, Pima Diabetes, UCI CKD) into a unified 7-biomarker schema across 2,088 patients and 6 non-IID hospital partitions with strictly zero data leakage.", CYAN),
        ("Objective 2: Multi-Task Federated Trunk", "Design a shared-trunk neural architecture (64 -> 32) predicting 3 chronic diseases simultaneously with masked BCE loss, and overcome institutional drift using FedProx (μ=0.01).", INDIGO),
        ("Objective 3: Formal Differential Privacy", "Enforce mathematically proven (ε, δ)-Differential Privacy via Opacus DP-SGD across budgets (ε ∈ {1.0, 3.0, 8.0, ∞}) with Rényi DP accounting at δ=10^-5.", EMERALD),
        ("Objective 4: Dual Subgroup Fairness Mitigations", "Formulate and audit dual fairness mitigations: in-processing client covariance loss penalties coupled with server-side FairFed aggregation reweighting, verified across 40 factorial trials.", ROSE)
    ]

    for idx, (title, desc, col) in enumerate(cards_data):
        row = idx // 2
        col_idx = idx % 2
        l = Inches(0.8) + col_idx * Inches(5.9)
        t = Inches(1.9) + row * Inches(2.4)
        add_card(s3, l, t, Inches(5.6), Inches(2.1), title, border_color=col)
        tb = s3.shapes.add_textbox(l + Inches(0.2), t + Inches(0.65), Inches(5.2), Inches(1.3))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = desc
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED
        p.space_before = Pt(2)

    # =========================================================================
    # SLIDE 4: DATA HARMONIZATION & 6 NON-IID HOSPITALS
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Data Engineering", "Harmonized Clinical Cohort & 6 Hospital Partitions", "Zero data leakage partitioning across 2,088 patient records")

    # Left: Feature schema
    add_card(s4, Inches(0.8), Inches(1.9), Inches(4.2), Inches(4.7), "Standardized Biomarkers (x ∈ R^7)", border_color=CYAN)
    tb_feat = s4.shapes.add_textbox(Inches(1.0), Inches(2.6), Inches(3.8), Inches(3.8))
    tf_f = tb_feat.text_frame
    feats = [
        "Age (years) [20 - 85]",
        "Biological Sex (0 = Female, 1 = Male)",
        "Body Mass Index (BMI, kg/m²)",
        "Systolic Blood Pressure (mmHg)",
        "Fasting Blood Glucose (mg/dL)",
        "Total Serum Cholesterol (mg/dL)",
        "Serum Creatinine (mg/dL)",
        "",
        "Multi-Task Targets (y ∈ {0, 1}^3):",
        "• Type 2 Diabetes Mellitus",
        "• Coronary Artery Disease (CAD)",
        "• Chronic Kidney Disease (CKD)"
    ]
    for i, f in enumerate(feats):
        p = tf_f.paragraphs[0] if i == 0 else tf_f.add_paragraph()
        p.text = f
        p.font.size = Pt(11)
        p.font.color.rgb = CYAN if "Targets" in f else TEXT_MUTED
        p.font.bold = True if "Targets" in f else False

    # Right: 6 Hospital Silos
    add_card(s4, Inches(5.3), Inches(1.9), Inches(7.2), Inches(4.7), "6 Non-IID Hospital Nodes (N=2,088)")
    tb_hosp = s4.shapes.add_textbox(Inches(5.5), Inches(2.6), Inches(6.8), Inches(3.8))
    tf_h = tb_hosp.text_frame
    hosps = [
        ("Hospital 0: Cleveland Clinic (CAD Specialty)", "N=242 train, 61 test | 70.2% Male | Obs: CAD, Diabetes"),
        ("Hospital 1: Hungarian Cardiology (Budapest)", "N=235 train, 59 test | 74.0% Male | Younger European CAD cohort"),
        ("Hospital 2: VA Long Beach & Alpine", "N=258 train, 65 test | 94.2% Male | Elderly high-severity veterans"),
        ("Hospital 3: Apollo Nephrology (CKD Specialty)", "N=240 train, 60 test | 67.1% Male | Renal profiling & GFR metrics"),
        ("Hospital 4: Southwest Metabolic (Pima)", "N=440 train, 110 test | 100% Female | Metabolic diabetes diagnostic center"),
        ("Hospital 5: Metro Community Primary Care", "N=254 train, 64 test | 22.0% Male | Multimorbid primary care blend")
    ]
    for idx, (h_name, h_desc) in enumerate(hosps):
        p = tf_h.paragraphs[0] if idx == 0 else tf_h.add_paragraph()
        p.text = "🏢 " + h_name
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_MAIN
        p2 = tf_h.add_paragraph()
        p2.text = "     " + h_desc
        p2.font.size = Pt(10)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_after = Pt(4)

    # =========================================================================
    # SLIDE 5: MULTI-TASK ARCHITECTURE & MASKED LOSS
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "Model Architecture", "Multi-Task Shared Trunk & Masked BCE Loss", "Simultaneous prediction of comorbid pathologies with partially observed tests")

    add_card(s5, Inches(0.8), Inches(1.9), Inches(5.7), Inches(4.7), "Shared Trunk Neural Network", border_color=INDIGO)
    tb_arch = s5.shapes.add_textbox(Inches(1.0), Inches(2.6), Inches(5.3), Inches(3.8))
    tf_a = tb_arch.text_frame
    tf_a.word_wrap = True
    arch_points = [
        "Biological Justification: Diabetes, CAD, and CKD share pathophysiological drivers (endothelial damage, hypertension, vascular disease).",
        "Layer 1: Input Layer (7 biomarkers) -> Dense 64 ReLU",
        "Layer 2: Shared Multi-Task Trunk -> Dense 32 ReLU",
        "Task Heads (3 Independent Linear + Sigmoid Heads):",
        "  • Head 1 (Diabetes Risk): 32 -> 1 (Sigmoid)",
        "  • Head 2 (Coronary CAD Risk): 32 -> 1 (Sigmoid)",
        "  • Head 3 (Kidney CKD Risk): 32 -> 1 (Sigmoid)",
        "Communication Efficiency: Transmits 1 shared trunk per FL round instead of 3 separate siloed networks."
    ]
    for i, pt in enumerate(arch_points):
        p = tf_a.paragraphs[0] if i == 0 else tf_a.add_paragraph()
        p.text = pt
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(6)

    add_card(s5, Inches(6.8), Inches(1.9), Inches(5.7), Inches(4.7), "Masked Multi-Task Loss Formulation", border_color=EMERALD)
    tb_loss = s5.shapes.add_textbox(Inches(7.0), Inches(2.6), Inches(5.3), Inches(3.8))
    tf_l = tb_loss.text_frame
    tf_l.word_wrap = True
    loss_points = [
        "Real-world clinical reality: Hospitals only test for modalities aligned with their clinical focus.",
        "Observation Mask M_k ∈ {0, 1}: Binary flag indicating whether disease k was ascertained.",
        "Class Imbalance Weight w_k = N_neg / N_pos: Compensates for positive-class skew.",
        "Mathematical Loss Equation:",
        "L_MT(y, y_hat, M) = Σ M_k · [ - w_k y_k log(y_hat_k) - (1 - y_k) log(1 - y_hat_k) ]",
        "Key Property: Gradients backpropagate only through valid clinical diagnostic evaluations without imputing phantom labels."
    ]
    for i, pt in enumerate(loss_points):
        p = tf_l.paragraphs[0] if i == 0 else tf_l.add_paragraph()
        p.text = pt
        p.font.size = Pt(11)
        p.font.color.rgb = EMERALD if "Loss Equation" in pt or "L_MT" in pt else TEXT_MUTED
        p.font.bold = True if "L_MT" in pt else False
        p.space_after = Pt(6)

    # =========================================================================
    # SLIDE 6: FEDERATED OPTIMIZATION (FEDAVG VS FEDPROX)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Federated Optimization", "Combating Institutional Drift with FedProx (μ=0.01)", "Constraining local model drift under severe non-IID label skew")

    # Metrics Grid
    col_w4 = Inches(2.7)
    gap4 = Inches(0.25)
    l_start4 = Inches(0.8)

    m_cards = [
        ("Centralized (Ideal)", "0.9143", "Macro AUROC (Pooled)", CYAN),
        ("Silo Collapse", "0.6072", "H2 VA Long Beach", ROSE),
        ("FedAvg (R=15)", "0.7572", "Drifts on non-IID", AMBER),
        ("FedProx (μ=0.01)", "0.8003", "+0.0431 gain over FedAvg", EMERALD)
    ]
    for i, (l_text, val_text, sub_text, col) in enumerate(m_cards):
        add_card(s6, l_start4 + i*(col_w4+gap4), Inches(1.9), col_w4, Inches(1.3), border_color=col)
        tb = s6.shapes.add_textbox(l_start4 + i*(col_w4+gap4) + Inches(0.15), Inches(2.0), col_w4 - Inches(0.3), Inches(1.1))
        p1 = tb.text_frame.paragraphs[0]
        p1.text = l_text
        p1.font.size = Pt(10)
        p1.font.color.rgb = TEXT_DIM
        p2 = tb.text_frame.add_paragraph()
        p2.text = val_text
        p2.font.size = Pt(22)
        p2.font.bold = True
        p2.font.color.rgb = col
        p3 = tb.text_frame.add_paragraph()
        p3.text = sub_text
        p3.font.size = Pt(9)
        p3.font.color.rgb = TEXT_MUTED

    # Bottom comparison explanation
    add_card(s6, Inches(0.8), Inches(3.5), Inches(11.7), Inches(3.2))
    tb_exp = s6.shapes.add_textbox(Inches(1.0), Inches(3.7), Inches(11.3), Inches(2.8))
    tf_e = tb_exp.text_frame
    tf_e.word_wrap = True
    exp_bullets = [
        "Why Local Silos Fail: Hospital 2 (VA Long Beach) achieved only 0.6072 AUROC and Hospital 4 (Pima) achieved 0.6155 AUROC on the multi-center test set due to severe institutional domain shift.",
        "Why FedAvg Struggles: Under extreme label skew, local gradients diverge in orthogonal directions. Averaging divergent weights destroys specialized feature filters.",
        "The FedProx Solution: Adds proximal regularizer (μ / 2) · ||θ - θ_global||^2 to local loss. Acts as an elastic attractor preventing local weights from wandering away from the global consensus trunk.",
        "Clinical Impact: FedProx boosted Macro AUROC from 0.7572 to 0.8003 (+0.0431 gain, +7.7% on CKD detection) and compressed Brier calibration error from 0.1836 to 0.1582."
    ]
    for i, b in enumerate(exp_bullets):
        p = tf_e.paragraphs[0] if i == 0 else tf_e.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED
        p.space_after = Pt(8)

    # =========================================================================
    # SLIDE 7: DIFFERENTIAL PRIVACY VIA OPACUS DP-SGD
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Differential Privacy", "Formal (ε, δ)-Differential Privacy via Opacus DP-SGD", "Bounding information disclosure with Rényi DP accounting (δ = 10^-5)")

    # Left: Explanation & Table
    add_card(s7, Inches(0.8), Inches(1.9), Inches(5.8), Inches(4.7), "DP-SGD Privacy Expenditure")
    tb_dp = s7.shapes.add_textbox(Inches(1.0), Inches(2.6), Inches(5.4), Inches(3.8))
    tf_dp = tb_dp.text_frame
    tf_dp.word_wrap = True
    dp_pts = [
        "1. Per-Sample Gradient Clipping: Clips individual gradient vectors to maximum L2 norm C = 1.0 to bound client influence.",
        "2. Gaussian Noise Injection: Injects N(0, σ² C² I) noise into summed gradients prior to federated communication.",
        "3. Rényi DP Accounting: Tight composition accounting over 15 rounds at δ = 10^-5.",
        "",
        "Empirical Privacy-Utility Tradeoff:",
        "• ε = 1.0 (Strict): Spent 0.95 | σ = 7.7856 | AUROC: 0.7182",
        "• ε = 3.0 (Balanced): Spent 2.84 | σ = 3.0037 | AUROC: 0.7536",
        "• ε = 8.0 (Permissive): Spent 7.56 | σ = 1.4509 | AUROC: 0.7724",
        "• ε = ∞ (Non-Private): No noise | σ = 0.0000 | AUROC: 0.8025"
    ]
    for i, pt in enumerate(dp_pts):
        p = tf_dp.paragraphs[0] if i == 0 else tf_dp.add_paragraph()
        p.text = pt
        p.font.size = Pt(11)
        p.font.color.rgb = CYAN if "AUROC:" in pt else TEXT_MUTED
        p.font.bold = True if "AUROC:" in pt else False
        p.space_after = Pt(3)

    # Right: Embed fig1_accuracy_vs_epsilon.png if exists
    p1_path = r"C:\Users\as\Desktop\FairFed Risk\results\plots\fig1_accuracy_vs_epsilon.png"
    if os.path.exists(p1_path):
        add_card(s7, Inches(6.9), Inches(1.9), Inches(5.6), Inches(4.7), "Accuracy vs. Privacy Budget (ε)", border_color=CYAN)
        s7.shapes.add_picture(p1_path, Inches(7.1), Inches(2.5), Inches(5.2), Inches(3.8))

    # =========================================================================
    # SLIDE 8: FAIRNESS & SUBGROUP EQUITY MITIGATIONS
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Algorithmic Fairness", "Subgroup Equity Auditing & Dual Mitigations", "Compressing demographic disparities across biological sex and age tiers")

    # Left: Embed fig2_accuracy_vs_fairness_gap.png
    p2_path = r"C:\Users\as\Desktop\FairFed Risk\results\plots\fig2_accuracy_vs_fairness_gap.png"
    if os.path.exists(p2_path):
        add_card(s8, Inches(0.8), Inches(1.9), Inches(5.6), Inches(4.7), "Accuracy vs. Fairness Gap (Pareto Frontier)", border_color=EMERALD)
        s8.shapes.add_picture(p2_path, Inches(1.0), Inches(2.5), Inches(5.2), Inches(3.8))

    # Right: Dual Mitigation Mechanics
    add_card(s8, Inches(6.7), Inches(1.9), Inches(5.8), Inches(4.7), "Dual Mitigation Architecture", border_color=ROSE)
    tb_fair = s8.shapes.add_textbox(Inches(6.9), Inches(2.6), Inches(5.4), Inches(3.8))
    tf_fr = tb_fair.text_frame
    tf_fr.word_wrap = True
    fair_bullets = [
        "Unmitigated Disparity: FedProx exhibits Sex DPD = 0.3047 and Sex EOD = 0.3409 due to male-heavy cardiac cohorts under-predicting female risk.",
        "Mitigation 1 (Client Covariance Penalty):",
        "  L_total = L_MT + λ_fair · Σ |Cov(a, y_hat_k)|",
        "  Penalizes statistical dependence between sensitive attribute a and predicted risk. At λ=0.6: Sex DPD drops to 0.1331 (56.3% reduction) with 98.3% utility retention.",
        "Mitigation 2 (FairFed Server Reweighting):",
        "  Server dynamically modulates aggregation weight β_k based on local fairness metric deviation from global consensus.",
        "Dual Pipeline Result: Sex DPD compressed to 0.0571 (81.3% reduction) and Age DPD to 0.2923."
    ]
    for i, b in enumerate(fair_bullets):
        p = tf_fr.paragraphs[0] if i == 0 else tf_fr.add_paragraph()
        p.text = b
        p.font.size = Pt(11)
        p.font.color.rgb = ROSE if "Dual Pipeline" in b else TEXT_MUTED
        p.font.bold = True if "Dual Pipeline" in b or "Mitigation" in b else False
        p.space_after = Pt(4)

    # =========================================================================
    # SLIDE 9: JOINT FACTORIAL STUDY MATRIX
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Joint Factorial Study", "Privacy × Fairness Interaction across 5 Seeds (40 Runs)", "Investigating the mathematical interplay between DP noise and fairness regularizers")

    p3_path = r"C:\Users\as\Desktop\FairFed Risk\results\plots\fig3_subgroup_disparity_under_dp.png"
    p4_path = r"C:\Users\as\Desktop\FairFed Risk\results\plots\fig4_per_disease_tradeoffs.png"
    
    if os.path.exists(p3_path) and os.path.exists(p4_path):
        add_card(s9, Inches(0.8), Inches(1.9), Inches(5.6), Inches(4.7), "Subgroup Disparity under DP-SGD", border_color=INDIGO)
        s9.shapes.add_picture(p3_path, Inches(1.0), Inches(2.5), Inches(5.2), Inches(3.8))

        add_card(s9, Inches(6.8), Inches(1.9), Inches(5.7), Inches(4.7), "Per-Disease Trade-offs (CAD vs CKD vs Diabetes)", border_color=CYAN)
        s9.shapes.add_picture(p4_path, Inches(7.0), Inches(2.5), Inches(5.3), Inches(3.8))

    # =========================================================================
    # SLIDE 10: 5-SEED EMPIRICAL BENCHMARK TABLE
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Empirical Findings", "Factorial Benchmark Table (Mean ± Std over 5 Seeds)", "Verified across seeds [42, 43, 44, 45, 46] tracked in SQLite MLflow")

    # Table Shape
    rows = 9
    cols = 6
    tbl_shape = s10.shapes.add_table(rows, cols, Inches(0.8), Inches(1.9), Inches(11.7), Inches(4.6))
    tbl = tbl_shape.table

    headers = ["Privacy (ε)", "Fairness Regime", "Macro AUROC", "Macro Brier", "Sex DPD", "Sex EOD"]
    table_data = [
        ["ε = 1.0", "No Mitigation", "0.6694 ± 0.0589", "0.2172 ± 0.0040", "0.0760 ± 0.0298", "0.0880 ± 0.0395"],
        ["ε = 1.0", "Fairness Mitigated", "0.6462 ± 0.0727", "0.2223 ± 0.0066", "0.0814 ± 0.0566", "0.0969 ± 0.0581"],
        ["ε = 3.0", "No Mitigation", "0.7090 ± 0.0443", "0.2125 ± 0.0044", "0.0764 ± 0.0289", "0.1275 ± 0.0342"],
        ["ε = 3.0", "Fairness Mitigated", "0.6736 ± 0.0814", "0.2181 ± 0.0054", "0.0773 ± 0.0205", "0.1412 ± 0.0301"],
        ["ε = 8.0", "No Mitigation", "0.7364 ± 0.0232", "0.2058 ± 0.0095", "0.0832 ± 0.0597", "0.1567 ± 0.0652"],
        ["ε = 8.0", "Fairness Mitigated", "0.6854 ± 0.0743", "0.2109 ± 0.0135", "0.0858 ± 0.0750", "0.1415 ± 0.0709"],
        ["ε = ∞", "No Mitigation", "0.7923 ± 0.0102", "0.1685 ± 0.0063", "0.2710 ± 0.0246", "0.3388 ± 0.0303"],
        ["ε = ∞", "Fairness Mitigated", "0.7531 ± 0.0118", "0.1947 ± 0.0031", "0.1546 ± 0.0433", "0.1917 ± 0.0377"]
    ]

    for c_idx, h in enumerate(headers):
        cell = tbl.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(2, 132, 199)
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_MAIN

    for r_idx, row in enumerate(table_data):
        is_highlight = (r_idx == 7)
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(16, 185, 129) if (is_highlight and c_idx >= 4) else (CARD_BG if r_idx % 2 == 0 else RGBColor(24, 36, 64))
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            p.font.color.rgb = TEXT_MAIN if (is_highlight and c_idx >= 4) else (CYAN if c_idx == 0 else TEXT_MUTED)
            p.font.bold = True if is_highlight else False

    # =========================================================
    # SLIDE 11: LIVE WEB SUITE & PATIENT SIMULATOR
    # =========================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(s11, "Demonstration", "Live Interactive Web Auditing Suite & Patient Simulator", "Hosted on Vercel at fairfed-risk.vercel.app with instant client-side execution")

    # 4 Feature Cards
    f_cards = [
        ("Executive Overview & Scrubber", "Interactive round scrubber (1-15) simulating Flower client gradient flow, FedProx proximal pullback, and institutional loss convergence.", CYAN),
        ("Differential Privacy Lab", "Real-time Gaussian noise density curve N(0, σ²) canvas dynamically reshaping as ε slider moves against the C=1.0 clipping sphere.", INDIGO),
        ("Subgroup Disparity Switcher", "Interactive attribute toggling between Biological Sex and Age tiers with DPD / EOD threshold indicators.", EMERALD),
        ("Patient Triage Engine & PDF", "Real-time multi-disease risk gauges with bias mitigation calibration toggle and print-ready clinical chart report export.", ROSE)
    ]
    for i, (f_title, f_desc, f_col) in enumerate(f_cards):
        x = Inches(0.8) + (i % 2) * Inches(5.9)
        y = Inches(1.9) + (i // 2) * Inches(2.4)
        add_card(s11, x, y, Inches(5.6), Inches(2.1), f_title, border_color=f_col)
        tb = s11.shapes.add_textbox(x + Inches(0.2), y + Inches(0.65), Inches(5.2), Inches(1.3))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = f_desc
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED

    # =========================================================
    # SLIDE 12: CODEBASE ARCHITECTURE
    # =========================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12)
    add_header(s12, "Software Engineering", "Modular Codebase Architecture & Test Verification", "23/23 Unit tests passing with strict separation of concerns")

    # Directory tree
    add_card(s12, Inches(0.8), Inches(1.9), Inches(5.6), Inches(4.7), "Modular Project Structure", border_color=CYAN)
    tb_tree = s12.shapes.add_textbox(Inches(1.0), Inches(2.6), Inches(5.2), Inches(3.8))
    tf_tr = tb_tree.text_frame
    tree_lines = [
        "src/data/      -> Harmonization, 7 features, non-IID partitions",
        "src/models/    -> MultiTaskTrunkMLP & Masked Multi-Task BCE",
        "src/federated/ -> Flower Client, FedProx proximal loss, Server",
        "src/privacy/   -> Opacus RDP accountant, noise calibration",
        "src/fairness/  -> DPD & EOD auditing, covariance loss penalty",
        "src/evaluation/-> AUROC, AUPRC, Brier score, calibration",
        "dashboard/     -> Standalone Web Suite (HTML/JS) & Streamlit",
        "tests/         -> 23 Comprehensive PyTest verification tests"
    ]
    for i, line in enumerate(tree_lines):
        p = tf_tr.paragraphs[0] if i == 0 else tf_tr.add_paragraph()
        p.text = line
        p.font.size = Pt(11)
        p.font.color.rgb = CYAN if "src/" in line else TEXT_MUTED
        p.space_after = Pt(4)

    # Unit Tests card
    add_card(s12, Inches(6.8), Inches(1.9), Inches(5.7), Inches(4.7), "Unit Test Suite (23 Tests Passing)", border_color=EMERALD)
    tb_tst = s12.shapes.add_textbox(Inches(7.0), Inches(2.6), Inches(5.3), Inches(3.8))
    tf_ts = tb_tst.text_frame
    tests_summary = [
        "✓ test_data.py: Verified 7 biomarkers, multi-task targets, masks",
        "✓ test_partition.py: Strictly verified ZERO data leakage across 6 sites",
        "✓ test_models.py: Verified shared trunk shape & masked loss flow",
        "✓ test_federated.py: Verified FedAvg, FedProx proximal pullback contraction",
        "✓ test_dp_accountant.py: Verified Opacus (ε, δ) bounds & noise calibration",
        "✓ test_fairness.py: Verified DPD, EOD parity & covariance regularizers",
        "",
        "CI Verification: pytest tests/ -> 23 passed in 20.38s"
    ]
    for i, t in enumerate(tests_summary):
        p = tf_ts.paragraphs[0] if i == 0 else tf_ts.add_paragraph()
        p.text = t
        p.font.size = Pt(11)
        p.font.color.rgb = EMERALD if "✓" in t or "CI" in t else TEXT_MUTED
        p.font.bold = True if "CI" in t else False
        p.space_after = Pt(4)

    # =========================================================
    # SLIDE 13: FUTURE SCOPE & LIMITATIONS
    # =========================================================
    s13 = prs.slides.add_slide(blank_layout)
    set_slide_background(s13)
    add_header(s13, "Future Scope", "Limitations & Clinical Translation Roadmap", "Pathways toward real-world deployment in distributed hospital networks")

    future_data = [
        ("Multi-Modal Deep Learning", "Integrating 12-lead electrocardiograms (ECG) and renal ultrasound scans alongside tabular electronic biomarkers using cross-attention multimodal transformers.", CYAN),
        ("Hardware Cryptographic Aggregation", "Combining DP-SGD with Secure Multi-Party Computation (SMPC) or Trusted Execution Environments (Intel SGX / ARM TrustZone) to eliminate cleartext gradient exposure to the aggregation server.", INDIGO),
        ("FHIR / HL7 Clinical Integration", "Packaging the patient risk inference engine as an open SMART-on-FHIR plugin compatible with major EHR systems (Epic, Cerner).", EMERALD),
        ("Prospective Clinical Trials", "Moving beyond open benchmark cohorts to prospective validation across live healthcare partner networks with regulatory compliance (FDA SaMD / CE Mark).", AMBER)
    ]
    for i, (title, desc, col) in enumerate(future_data):
        x = Inches(0.8) + (i % 2) * Inches(5.9)
        y = Inches(1.9) + (i // 2) * Inches(2.4)
        add_card(s13, x, y, Inches(5.6), Inches(2.1), title, border_color=col)
        tb = s13.shapes.add_textbox(x + Inches(0.2), y + Inches(0.65), Inches(5.2), Inches(1.3))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = desc
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MUTED

    # =========================================================
    # SLIDE 14: CONCLUSION & Q&A
    # =========================================================
    s14 = prs.slides.add_slide(blank_layout)
    set_slide_background(s14)

    # Decorative header
    add_card(s14, Inches(0.8), Inches(1.5), Inches(11.7), Inches(4.8), border_color=CYAN)
    tb_c = s14.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(11.0), Inches(4.2))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True

    p = tf_c.paragraphs[0]
    p.text = "Conclusion & Research Summary"
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN

    concl_points = [
        "1. Solved Institutional Silo Collapse: FedProx multi-task learning achieves 0.8003 AUROC across 6 non-IID hospitals, outperforming isolated silos (0.607-0.785) and FedAvg (0.7572).",
        "2. Formally Bounded Privacy: Opacus DP-SGD guarantees rigorous (ε, δ=10^-5) privacy bounds across rounds.",
        "3. Mitigated Subgroup Disparity: Dual mitigation compresses biological sex demographic disparity by 81.3% while retaining 98.3% predictive utility.",
        "4. Scientifically Audited: 40 factorial trials across 5 seeds with interactive open-source web auditability.",
        "",
        "Thank you respected Guide, Head of Department, and Evaluation Committee Members.",
        "I am now open to your questions and discussion."
    ]
    for pt in concl_points:
        p_pt = tf_c.add_paragraph()
        p_pt.text = pt
        p_pt.font.size = Pt(13)
        p_pt.font.color.rgb = CYAN if "Thank you" in pt or "open to your questions" in pt else TEXT_MUTED
        p_pt.font.bold = True if "Thank you" in pt else False
        p_pt.space_after = Pt(6)

    # Save presentation
    output_path = r"C:\Users\as\Desktop\FairFed Risk\FairFed_Risk_MTech_Presentation.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to {output_path}")

    # Export PDF backup using PowerPoint COM
    pdf_path = r"C:\Users\as\Desktop\FairFed Risk\FairFed_Risk_MTech_Presentation.pdf"
    try:
        import win32com.client
        import time
        print("Exporting PDF backup via Microsoft PowerPoint...")
        powerpoint = win32com.client.Dispatch("PowerPoint.Application")
        deck = powerpoint.Presentations.Open(os.path.abspath(output_path), WithWindow=False)
        # 32 = ppSaveAsPDF
        deck.SaveAs(os.path.abspath(pdf_path), 32)
        deck.Close()
        powerpoint.Quit()
        print(f"PDF backup exported successfully to {pdf_path}")
    except Exception as e:
        print(f"Note on PDF export: {e}")

if __name__ == "__main__":
    create_deck()
