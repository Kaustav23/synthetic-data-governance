# 🛡️ Synthetic Data Quality & Privacy Governance Terminal

**Author:** Kaustav Goswami ([GitHub: Kaustav23](https://www.google.com/search?q=https://github.com/Kaustav23))  
**Domain:** Enterprise Data Privacy Governance, Differential Privacy, & Clinical BI Operations  
**Tech Stack:** Python, Streamlit, Pandas, NumPy, SciPy, Plotly

An interactive, research-grounded governance terminal that evaluates a restricted clinical dataset containing Personally Identifiable Information (PII) against a privacy-preserving **Synthetic Twin**. Built for Chief Data Officers, Data Privacy Architects, IRB/Compliance Officers, and Analytics Leaders, the application pairs rigorous mathematical validation with plain-English executive translations.

---

## 🎯 Project Vision & Business Problem

Enterprise analytics and AI teams face a structural bottleneck: raw clinical and financial datasets contain direct identifiers (names, SSNs, Medical Record Numbers) and quasi-identifiers (age, ZIP code, blood pressure, cholesterol) that can be combined to re-identify real individuals. According to the [Perforce Delphix 2026 Test Data Management Survey (via DataCebo)](https://datacebo.com/blog/capabilities-of-synthetic-test-data/), **99% of enterprises wait more than 1 business day** for compliant production data copies, and **42% wait weeks or months** while requests sit in legal, IRB, and database administrator queues.

This dashboard resolves that bottleneck by auditing a **Gaussian Copula + Differential Privacy (`ε`) Synthetic Twin** across three core governance pillars before unlocking self-service access in **30 minutes (`0.5 hours`)**:

1. **Statistical Parity Score (`97.71%` at `ε = 1.0`):** *"Does this fake data look and behave like the real data?"*  
   Evaluates marginal distribution shapes via [SDMetrics KSComplement (`1 - D_KS`)](https://docs.sdv.dev/sdmetrics/data-metrics/quality/kscomplement) for continuous variables, [TVComplement (`1 - TVD`)](https://docs.sdv.dev/sdmetrics/data-metrics/quality/tvcomplement) for categorical variables, and pairwise [CorrelationSimilarity](https://docs.sdv.dev/sdmetrics/data-metrics/quality/correlationsimilarity) in the [SDMetrics Quality Report](https://docs.sdv.dev/sdmetrics/data-metrics/quality/quality-report-api), benchmarked against the `85.53%` empirical column-shape fidelity standard published in [Frontiers in Sports and Active Living (2025)](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1607600/full).
2. **PII Leakage Risk (`1.28%` at `ε = 1.0`):** *"What is the mathematical probability that a hacker could reverse-engineer a real person's identity from this fake data?"*  
   Measures meaningful identity disclosure risk against the **`0.09` (`9.0%`, `k >= 11` k-anonymity)** regulatory safe-harbor ceiling mandated by [EMA Policy 0070 and Health Canada (PMC7029478)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7029478/) and validated on synthetic health cohorts (`0.86%–1.98%` risk) by [El Emam, Mosquera & Bass (2020) in *JMIR*](https://www.jmir.org/2020/11/e23139/), alongside worst-case Membership Inference Attack (MIA) bounds (`e^ε / (1 + e^ε)`) under [NIST SP 800-226 (March 2025)](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-226.pdf) and [Differential Privacy Theory](https://www.ultralytics.com/glossary/differential-privacy).
3. **Data Democratization Rate (`240x` Speedup):** *"How much faster can our analysts start working?"*  
   Quantifies engineering velocity and annual cost savings when compressing a **3-week (`120 work-hour`)** raw PII legal/IRB review cycle ([Enterprise Synthetic Data Benchmark](https://medium.com/@reliabledataengineering/synthetic-data-the-100k-solution-data-teams-are-sleeping-on-while-burning-money-on-real-data-f0ecf06a0e69)) down to a **30-minute (`0.5-hour`)** automated synthetic provisioning pipeline ([Perforce Greenfield Synthetic Data Benchmark](https://www.perforce.com/blog/pdx/synthetic-for-greenfield-app-dev)).

---

## 🏗️ System Architecture & Data Pipeline

1. **Restricted Raw PII Cohort (`raw_pii_data`):**  
   Simulates an enterprise Electronic Health Record (EHR) cohort grounded in the empirical parameters of the [Framingham Heart Study Hypertension ML Analysis (JBS, n=4,240)](https://iauejournals.com/index.php/JBS/article/download/106/106), the [Framingham Healthcare Analytics Case Study](https://www.researchgate.net/publication/336703415_Healthcare_Analytics_A_Case_Study_Approach_Using_the_Framingham_Heart_Study), the [Fasa Cohort (PMC11194982, n=8,944)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11194982/), and the [Cardiovascular Metabolic Disorders Cohort](https://pdfs.semanticscholar.org/c64b/eae97fb5e7a90f314319eeaa54670355059b.pdf). Includes direct identifiers (`patient_mrn`, `full_name`, `ssn`, `exact_dob`, `zip_code`) alongside correlated clinical features.
2. **Differential Privacy Copula Engine (`data_governance_engine.py`):**  
   Fits a multivariate Gaussian Copula covariance matrix preserving the empirical `r = 0.787` correlation between Systolic and Diastolic Blood Pressure, projects the DP-perturbed covariance onto the nearest positive-semidefinite matrix via eigendecomposition, injects calibrated Laplace/Gaussian noise scaled by `1 / ε`, strips all direct PII, and generalizes geographic regions to 3-digit ZIP prefixes.
3. **Interactive Governance Views (`app.py`):**
   - **View 1 — Executive Governance Overview:** Real-time Safe-Harbor compliance verdict banner, Plain-English executive primer, regulatory gauge chart against the `9.0%` EMA/Health Canada ceiling, and horizontal feature parity bars.
   - **View 2 — Statistical Parity & Distribution Lab:** Overlaid probability density histograms, boxplots, Empirical Cumulative Distribution Functions (ECDF) showing the exact Kolmogorov-Smirnov distance (`D_KS`), and side-by-side multivariate correlation heatmaps.
   - **View 3 — Privacy Risk vs. Data Utility Frontier:** Interactive Pareto trade-off curve across Differential Privacy budgets `ε ∈ [0.1, 10.0]` and Membership Inference Attack (MIA) vulnerability diagnostics.
   - **View 4 — Restricted PII vs. Synthetic Safe Zone:** Side-by-side architectural inspection comparing restricted EHR rows against non-linkable synthetic twin records, plus a 4-stage compliance bottleneck ROI calculator.
   - **View 5 — Research Methodology & Citations:** Expandable audit registry linking every formula and threshold to its primary publication.

---

## 📊 Grounded Empirical Benchmarks at a Glance

| Governance Dimension | Metric / Parameter | Empirical Benchmark / Threshold | Primary Research Source |
| :--- | :--- | :--- | :--- |
| **Statistical Fidelity** | Continuous Feature Parity | `KSComplement = 1 - D_KS` (`> 70%`) | [SDMetrics KSComplement Documentation](https://docs.sdv.dev/sdmetrics/data-metrics/quality/kscomplement) |
| **Statistical Fidelity** | Categorical Feature Parity | `TVComplement = 1 - TVD` (`> 70%`) | [SDMetrics TVComplement Documentation](https://docs.sdv.dev/sdmetrics/data-metrics/quality/tvcomplement) |
| **Statistical Fidelity** | Published Column Shapes Score | `85.53%` Composite Fidelity | [Frontiers in Sports & Active Living (2025)](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1607600/full) |
| **Privacy Safe Harbor** | Max Acceptable Re-Identification Risk | `≤ 0.09` (`9.0%`, `k ≥ 11` k-anonymity) | [EMA Policy 0070 & Health Canada (PMC7029478)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7029478/) |
| **Privacy Safe Harbor** | Synthetic Health Disclosure Risk | `0.0086` (`0.86%`) to `0.0198` (`1.98%`) | [El Emam, Mosquera & Bass (2020), *JMIR*](https://www.jmir.org/2020/11/e23139/) |
| **Differential Privacy** | Worst-Case MIA Success Bound | `e^ε / (1 + e^ε)` (`52.5%` at `ε=0.1`, `73.1%` at `ε=1.0`) | [NIST SP 800-226 (March 2025)](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-226.pdf) |
| **Data Democratization** | Raw PII Approval Wait vs. Synthetic | `2–4 Weeks` (`42%` wait weeks/months) vs. `0.5 Hours` | [Perforce Delphix 2026 TDM Report (DataCebo)](https://datacebo.com/blog/capabilities-of-synthetic-test-data/) |
| **Clinical Cohort** | Framingham Baseline Parameters | Age `49.6±9.0`, SysBP `132.35±22.0`, DiaBP `82.90±11.7` (`r=0.787`) | [Framingham Hypertension ML Study (JBS)](https://iauejournals.com/index.php/JBS/article/download/106/106) |

---

## 📂 Repository Structure

```text
Synthetic-Data-Governance/
├── app.py                        # Main Streamlit governance dashboard (5 interactive views)
├── data_governance_engine.py     # DP-Copula synthetic generator & statistical/privacy evaluators
├── requirements.txt              # Production Python dependencies
├── README.md                     # Project overview, architecture, and deployment guide
└── METHODOLOGY_AND_CITATIONS.md  # Mathematical derivations and peer-reviewed reference manual
```

---

## 🚀 Local Installation & Execution (Windows PowerShell / VS Code)

1. **Navigate to your local project directory:**
   ```powershell
   cd D:\Project\Synthetic-Data-Governance
   ```

2. **Install dependencies:**
   ```powershell
   py -m pip install -r requirements.txt
   ```

3. **Launch the Streamlit governance dashboard:**
   ```powershell
   py -m streamlit run app.py
   ```

---

## ☁️ Deploying to GitHub & Streamlit Community Cloud

1. **Initialize Git and push to GitHub (`Kaustav23`):**
   ```powershell
   git init
   git add app.py data_governance_engine.py requirements.txt README.md METHODOLOGY_AND_CITATIONS.md
   git commit -m "Initial commit: Research-grounded Synthetic Data Quality & Privacy Governance Dashboard"
   git branch -M main
   git remote add origin https://github.com/Kaustav23/Synthetic-Data-Governance.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Sign in to [share.streamlit.io](https://share.streamlit.io/) with your `Kaustav23` GitHub account.
   - Select **New app**, choose repository `Kaustav23/Synthetic-Data-Governance`, branch `main`, and main file path `app.py`.
   - Click **Deploy**.
