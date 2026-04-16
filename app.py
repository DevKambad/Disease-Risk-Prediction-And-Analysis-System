import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import os
import traceback

st.set_page_config(page_title="Disease Risk Prediction System", layout="wide")

st.markdown("""
<style>

/* Background */
.main {
    background-color: #f5f7fb;
}

/* Cards */
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Titles */
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 10px;
}

/* Risk Colors */
.high-risk {
    color: #e53935;
    font-weight: bold;
}

.medium-risk {
    color: #fb8c00;
    font-weight: bold;
}

.low-risk {
    color: #43a047;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

st.write("Loading models...")
try:
    diabetes_pipeline = joblib.load("models/diabetes_pipeline.pkl")
    st.success("Diabetes model loaded successfully")
except Exception as e:
    st.error("Diabetes model loading failed")
    st.text(str(e))
    st.text(traceback.format_exc())

diabetes_threshold = joblib.load("models/diabetes_threshold.pkl")

try:
    heart_pipeline = joblib.load("models/heart_pipeline.pkl")
    st.success("Heart model loaded successfully")
except Exception as e:
    st.error("Heart model loading failed")
    st.text(str(e))
    st.text(traceback.format_exc())

heart_threshold = joblib.load("models/heart_threshold.pkl")


if "diabetes_result" not in st.session_state:
    st.session_state.diabetes_result = None

if "heart_result" not in st.session_state:
    st.session_state.heart_result = None

smoking_map = {
    "never": 0,
    "former": 1,
    "current": 2,
    "not current": 3,
    "ever": 4,
    "No Info": 5
}

cp_map = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

fbs_map = {
    "No (<120 mg/dl)": 0,
    "Yes (>120 mg/dl)": 1
}

restecg_map = {
    "Normal": 0,
    "ST-T Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

exang_map = {
    "No": 0,
    "Yes": 1
}

slope_map = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

ca_map = {
    "0 vessels": 0,
    "1 vessel": 1,
    "2 vessels": 2,
    "3 vessels": 3,
    "4 vessels": 4
}

thal_map = {
    "Normal": 0,
    "Fixed Defect": 1,
    "Reversible Defect": 2,
    "Unknown": 3
}

def get_diabetes_ranges():

    return {
        "bmi": {
            "underweight": (0, 18.5),
            "normal": (18.5, 24.9),
            "overweight": (25, 29.9),
            "obese": (30, 100)
        },
        "hba1c": {
            "normal": (0, 5.6),
            "prediabetes": (5.7, 6.4),
            "diabetes": (6.5, 20)
        },
        "glucose_fasting": {
            "normal": (70, 99),
            "prediabetes": (100, 125),
            "diabetes": (126, 300)
        }
    }

def get_heart_ranges():

    return {
        "bp": {
            "normal": (0, 119),
            "elevated": (120, 129),
            "hypertension": (130, 200)
        },
        "chol": {
            "normal": (0, 199),
            "borderline": (200, 239),
            "high": (240, 600)
        },
        "oldpeak": {
            "normal": (0.0, 0.9),
            "mild": (1.0, 1.9),
            "severe": (2.0, 6.0)
        }
    }


def get_risk_category(prob):
    if prob < 0.30:
        return "Low Risk", "green"
    elif prob < 0.70:
        return "Moderate Risk", "orange"
    else:
        return "High Risk", "red"

def apply_clinical_override(user_input, prob, disease):

    adjusted_prob = prob
    reasons = []

    if disease == "diabetes":
        glucose = user_input.get("blood_glucose_level", 0)
        hba1c = user_input.get("HbA1c_level", 0)
        bmi = user_input.get("bmi", 0)

        if glucose >= 126:
            adjusted_prob += 0.15
            reasons.append("High fasting glucose (≥126 mg/dL)")

        if hba1c >= 6.5:
            adjusted_prob += 0.15
            reasons.append("HbA1c ≥6.5%")

        if bmi >= 35:
            adjusted_prob += 0.15
            reasons.append("Severe obesity (BMI ≥35)")
    
    elif disease == "heart":
        bp = user_input.get("trestbps", 0)
        chol = user_input.get("chol", 0)
        oldpeak = user_input.get("oldpeak", 0)

        if bp >= 140:
            adjusted_prob += 0.10
            reasons.append("High blood pressure")

        if chol >= 240:
            adjusted_prob += 0.10
            reasons.append("High cholesterol")

        if oldpeak > 2:
            adjusted_prob += 0.15
            reasons.append("ST depression indicates cardiac stress")

    adjusted_prob = min(adjusted_prob, 1.0)

    return adjusted_prob, reasons

def explain_risk(prob):

    if prob < 0.30:
        return "Low predicted risk. This means the model estimates a low likelihood of disease based on current inputs."
    
    elif prob < 0.70:
        return "Moderate predicted risk. Some risk factors are present. Preventive measures are recommended."
    
    else:
        return "High predicted risk. Multiple risk factors detected. Clinical consultation is strongly advised."

        
def show_result(prob):
    label, color = get_risk_category(prob)
    percentage = prob * 100

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Clinical Risk Assessment</div>', unsafe_allow_html=True)

    st.metric("Risk Probability", f"{percentage:.2f}%")
    st.progress(float(prob))
    
    if color == "green":
        st.success(label)
    elif color == "orange":
        st.warning(label)
    else:
        st.error(label)

    st.markdown(
        f"<p style='text-align:center; color:gray;'>{explain_risk(prob)}</p>",
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)


def generate_clinical_explanation(user_input, prob, disease):

    explanation = ""

    if disease == "diabetes":
        bmi = user_input.get("bmi", 0)
        hba1c = user_input.get("HbA1c_level", 0)
        glucose = user_input.get("blood_glucose_level", 0)

        abnormal = []
        
        ranges = get_diabetes_ranges()
        
        if bmi >= ranges["bmi"]["overweight"][0]:
            abnormal.append(f"elevated BMI ({bmi})")
            
        if hba1c >= ranges["hba1c"]["prediabetes"][0]:
            abnormal.append(f"elevated HbA1c ({hba1c})")
            
        if glucose >= ranges["glucose_fasting"]["prediabetes"][0]:
            abnormal.append(f"elevated blood glucose ({glucose})")

        if abnormal:
            explanation += "The patient presents with " + ", ".join(abnormal)
            explanation += ", which are significant risk factors for diabetes. "
            
        if prob > 0.7:
            explanation += "Overall findings indicate a high likelihood of diabetes and require medical attention."
        elif prob > 0.3:
            explanation += "Moderate risk detected; lifestyle modifications are strongly recommended."
        else:
            explanation += "Risk appears low, but preventive care is advised."

    elif disease == "heart":
        bp = user_input.get("trestbps", 0)
        chol = user_input.get("chol", 0)
        oldpeak = user_input.get("oldpeak", 0)

        abnormal = []
        
        ranges = get_heart_ranges()
        
        if bp >= ranges["bp"]["hypertension"][0]:
            abnormal.append(f"high blood pressure ({bp})")
        elif bp >= ranges["bp"]["elevated"][0]:
            abnormal.append(f"elevated blood pressure ({bp})")
            
        if chol >= ranges["chol"]["high"][0]:
            abnormal.append(f"high cholesterol ({chol})")
        elif chol >= ranges["chol"]["borderline"][0]:
            abnormal.append(f"borderline cholesterol ({chol})")
            
        if oldpeak >= ranges["oldpeak"]["severe"][0]:
            abnormal.append(f"significant ST depression ({oldpeak})")
        elif oldpeak >= ranges["oldpeak"]["mild"][0]:
            abnormal.append(f"mild ST depression ({oldpeak})")
       
        if abnormal:
            explanation += "The patient presents with " + ", ".join(abnormal)
            explanation += ", which are associated with cardiovascular risk. "

        if prob > 0.7:
            explanation += "Findings suggest a high risk of heart disease and require clinical evaluation."
        elif prob > 0.3:
            explanation += "Moderate cardiovascular risk detected; monitoring and lifestyle changes advised."
        else:
            explanation += "Low cardiovascular risk, maintain healthy habits."

    return explanation

def show_clinical_explanation(user_input, prob, disease):

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Clinical Interpretation</div>', unsafe_allow_html=True)

    explanation = generate_clinical_explanation(user_input, prob, disease)

    st.info(explanation)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recommended Next Steps</div>', unsafe_allow_html=True)

    if prob > 0.7:
        st.error("Immediate consultation with a healthcare professional is advised.")
    elif prob > 0.3:
        st.warning("Adopt lifestyle changes and monitor health regularly.")
    else:
        st.success("Maintain current healthy lifestyle.")
    st.markdown('</div>', unsafe_allow_html=True)


def get_feature_names(pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        
        if name == "num":
            feature_names.extend(columns)

        elif name == "cat":
            ohe = transformer
            encoded_names = ohe.get_feature_names_out(columns)
            feature_names.extend(encoded_names)

    return feature_names


def show_feature_importance(pipeline):
    try:
        model = pipeline.named_steps["model"]
        feature_names = get_feature_names(pipeline)

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])

        else:
            st.info("Feature importance not available")
            return

        if len(feature_names) != len(importances):
            st.warning("Feature mismatch after preprocessing")
            return
            
        labels = get_feature_labels()
        
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        
        importance_df["Feature"] = importance_df["Feature"].map(labels).fillna(importance_df["Feature"])

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Key Risk Drivers</div>', unsafe_allow_html=True)
        st.bar_chart(importance_df.set_index("Feature").head(8))

        top_features = importance_df.head(3)["Feature"].tolist()

        st.markdown(
            f"**Top contributing factors:** {', '.join(top_features)}"
        )

    except Exception as e:
        st.warning("Feature importance unavailable")
        st.write(str(e))
        st.markdown('</div>', unsafe_allow_html=True)


def show_clinical_report(user_input, disease):

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Clinical Findings</div>', unsafe_allow_html=True)

    if disease == "diabetes":
        features = {
            "bmi": (18.5, 24.9, "Body Mass Index"),
            "HbA1c_level": (4.0, 5.6, "HbA1c (Normal < 5.7%)"),
            "blood_glucose_level": (70, 99, "Fasting Blood Glucose (Normal < 100 mg/dl)")
        }

    elif disease == "heart":
        features = {
            "trestbps": (90, 120, "Resting Blood Pressure (Normal < 120)"),
            "chol": (125, 200, "Cholesterol (Desirable < 200)"),
            "oldpeak": (0.0, 1.0, "ST Depression (Normal ~0)")
        }

    for key, (low, high, label) in features.items():

        value = user_input.get(key)

        if value is None:
            continue

        col1, col2 = st.columns([1, 2])

        if value < low:
            status = "Low"
            color = ""
        elif value > high:
            status = "High"
            color = ""
        else:
            status = "Normal"
            color = ""

        with col1:
            st.metric(label, f"{value}")

        with col2:
            st.markdown(f"""
            **Status:** {color} {status}  
            **Normal Range:** {low} – {high}
            """)

            if status == "High":
                st.error(f"{label} is above normal range and may increase disease risk.")
            elif status == "Low":
                st.warning(f"{label} is below normal range and may indicate abnormal condition.")
            else:
                st.success(f"{label} is within healthy range.")
        st.markdown('</div>', unsafe_allow_html=True)

def get_feature_ranges(disease):
    
    if disease == "diabetes":
        return {
            "bmi": [30, 27, 25, 23],
            "HbA1c_level": [7.0, 6.5, 6.0, 5.5],
            "blood_glucose_level": [140, 120, 105, 95]
        }

    elif disease == "heart":
        return {
            "chol": [240, 220, 200, 180],
            "thalach": [120, 140, 160, 180],
            "oldpeak": [3.0, 2.0, 1.0, 0.5]
        }

    return {}

def generate_counterfactuals(user_input, pipeline, disease):

    input_df = pd.DataFrame([user_input])

    raw_prob = pipeline.predict_proba(input_df)[0][1]
    base_prob, _ = apply_clinical_override(user_input, raw_prob, disease)

    feature_ranges = get_feature_ranges(disease)

    suggestions = []

    for feature, values in feature_ranges.items():

        if feature not in user_input:
            continue

        current_value = user_input[feature]

        best_value = current_value
        best_prob = base_prob

        feature_directions = {
            "bmi": "decrease",
            "HbA1c_level": "decrease",
            "blood_glucose_level": "decrease",
            "chol": "decrease",
            "oldpeak": "decrease",
            "thalach": "increase"
        }
        
        direction = feature_directions.get(feature, "decrease")
        
        for val in values:
            
            if direction == "decrease" and val >= current_value:
                continue
            if direction == "increase" and val <= current_value:
                continue

            new_input = user_input.copy()
            new_input[feature] = val

            new_df = pd.DataFrame([new_input])

            new_raw = pipeline.predict_proba(new_df)[0][1]
            new_prob, _ = apply_clinical_override(new_input, new_raw, disease)

            if new_prob < best_prob:
                best_prob = new_prob
                best_value = val

        improvement = base_prob - best_prob

        if improvement > 0 and best_value != current_value:

            suggestions.append({
                "feature": feature,
                "label": feature.replace("_", " ").title(),
                "current": current_value,
                "optimal": best_value,
                "improvement": improvement
            })

    suggestions = sorted(suggestions, key=lambda x: x["improvement"], reverse=True)

    return suggestions, base_prob

def show_counterfactuals(pipeline, user_input, disease):

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Risk Reduction Insights</div>', unsafe_allow_html=True)

    suggestions, base_prob = generate_counterfactuals(user_input, pipeline, disease)

    if suggestions:

        best = suggestions[0]

        st.markdown("##### Most Impactful Change")
        
        st.success(
            f"{best['label']}: {best['current']} → {best['optimal']}  \n"
            f"**Estimated Risk Reduction: ↓ {best['improvement']*100:.2f}%**"

        )

        st.info(
            f"If you improve **{best['label']}**, your overall risk can reduce significantly."
        )
        
        if len(suggestions) > 1:
            st.markdown("##### Other Recommendations")

            for s in suggestions[1:]:
                st.markdown(
                    f"• **{s['label']}**: {s['current']} → {s['optimal']} "
                    f"(↓ {s['improvement']*100:.2f}%)"
                )

    else:
        st.warning(
            "No significant risk-reducing changes identified. "
            "Maintain healthy habits and consult a healthcare professional."
        )

    st.markdown('</div>', unsafe_allow_html=True)

def format_value(key, value, disease):

    if disease == "diabetes":

        mappings = {
            "gender": {0: "Male", 1: "Female"},
            "hypertension": {0: "No", 1: "Yes"},
            "heart_disease": {0: "No", 1: "Yes"},
            "smoking_history": {
                0: "Never",
                1: "Former",
                2: "Current",
                3: "Not Current",
                4: "Ever",
                5: "No Info"
            }
        }

    elif disease == "heart":

        mappings = {
            "gender": {1: "Male", 0: "Female"},
            "cp": {
                0: "Typical Angina",
                1: "Atypical Angina",
                2: "Non-anginal Pain",
                3: "Asymptomatic"
            },
            "fbs": {0: "No", 1: "Yes"},
            "restecg": {
                0: "Normal",
                1: "ST-T Abnormality",
                2: "Left Ventricular Hypertrophy"
            },
            "exang": {0: "No", 1: "Yes"},
            "slope": {
                0: "Upsloping",
                1: "Flat",
                2: "Downsloping"
            },
            "ca": {
                0: "0 vessels",
                1: "1 vessel",
                2: "2 vessels",
                3: "3 vessels",
                4: "4 vessels"
            },
            "thal": {
                0: "Normal",
                1: "Fixed Defect",
                2: "Reversible Defect",
                3: "Unknown"
            }
        }

    else:
        mappings = {}

    if key in mappings and value in mappings[key]:
        return mappings[key][value]

    return value

def get_feature_labels():

    return {
        "age": "Age",
        "gender": "Gender",
        "bmi": "Body Mass Index",
        "HbA1c_level": "HbA1c (%)",
        "blood_glucose_level": "Blood Glucose (mg/dL)",
        "hypertension": "Hypertension",
        "heart_disease": "Heart Disease History",
        "smoking_history": "Smoking History",

        "cp": "Chest Pain Type",
        "trestbps": "Resting Blood Pressure",
        "chol": "Cholesterol",
        "fbs": "Fasting Blood Sugar",
        "restecg": "Rest ECG",
        "thalach": "Max Heart Rate",
        "exang": "Exercise Angina",
        "oldpeak": "ST Depression",
        "slope": "Slope",
        "ca": "Major Vessels",
        "thal": "Thalassemia"
    }

def get_feature_status(key, value, disease):

    if disease == "diabetes":

        if key == "bmi":
            if value < 18.5:
                return "Low", "orange"
            elif value <= 24.9:
                return "Normal", "green"
            elif value <= 29.9:
                return "Overweight", "orange"
            else:
                return "Obese", "red"

        elif key == "HbA1c_level":
            if value < 5.7:
                return "Normal", "green"
            elif value <= 6.4:
                return "Prediabetes", "orange"
            else:
                return "Diabetes", "red"

        elif key == "blood_glucose_level":
            if value < 100:
                return "Normal", "green"
            elif value <= 125:
                return "Prediabetes", "orange"
            else:
                return "Diabetes", "red"

    elif disease == "heart":

        if key == "trestbps":
            if value < 120:
                return "Normal", "green"
            elif value <= 129:
                return "Elevated", "orange"
            else:
                return "High", "red"

        elif key == "chol":
            if value < 200:
                return "Normal", "green"
            elif value <= 239:
                return "Borderline", "orange"
            else:
                return "High", "red"

        elif key == "oldpeak":
            if value < 1.0:
                return "Normal", "green"
            elif value <= 2.0:
                return "Mild", "orange"
            else:
                return "Severe", "red"

    return None, None

def create_risk_chart(prob):
    fig, ax = plt.subplots()
    ax.barh(["Risk"], [prob])
    ax.set_xlim(0,1)
    ax.set_title("Risk Probability")

    path = os.path.abspath("risk_chart.png")
    
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    
    return path

def create_feature_importance_chart(pipeline, feature_names):

    model = pipeline.named_steps['model']

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])

    indices = np.argsort(importances)[-5:]

    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(6, 3))
    plt.barh(top_features, top_importances)
    plt.xlabel("Importance")
    plt.title("Top Risk Drivers")

    chart_path = os.path.abspath("feature_importance.png")
    
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    return chart_path

def generate_pdf_report(user_input, prob, disease):

    file_name = os.path.abspath(f"{disease}_report.pdf")

    doc = SimpleDocTemplate(
        file_name,
        rightMargin=30, leftMargin=30,
        topMargin=30, bottomMargin=20
    )

    styles = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle(
        name="TitleStyle",
        fontSize=20,
        leading=24,
        alignment=1,
        textColor=colors.darkblue,
        spaceAfter=10
    )

    section_style = ParagraphStyle(
        name="SectionStyle",
        fontSize=14,
        leading=18,
        textColor=colors.black,
        spaceBefore=10,
        spaceAfter=6
    )

    normal_style = styles["Normal"]

    footer_style = ParagraphStyle(
        name="FooterStyle",
        fontSize=8,
        alignment=1,
        textColor=colors.grey
    )

    elements.append(Paragraph("Disease Risk Prediction Report", title_style))
    elements.append(Spacer(1,5))
    elements.append(Paragraph(
        "<i>Generated by Clinical Decision Support System</i>",
        styles["Italic"]
    ))
    elements.append(Paragraph(f"<b>Disease:</b> {disease.capitalize()}", normal_style))
    elements.append(Paragraph(
        f"<b>Generated on:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        normal_style
    ))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("1. Patient Summary", section_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 8))

    labels = get_feature_labels()

    if disease == "diabetes":
        feature_order = [
            "gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"
        ]
    else:
        feature_order = [
            "age", "gender", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak",
            "slope", "ca", "thal"
        ]

    for key in feature_order:

        value = user_input.get(key, None)

        if value is None:
            readable_value = "N/A"
        else:
            readable_value = format_value(key, value, disease)

        label = labels.get(key, key)

        status, color = get_feature_status(key, value, disease) if value is not None else (None, None)

        if status:
            text = f"<b>{label}:</b> {readable_value} <font color='{color}'>({status})</font>"
        else:
            text = f"<b>{label}:</b> {readable_value}"

        elements.append(Paragraph(text, normal_style))
        elements.append(Spacer(1, 5))

    elements.append(Spacer(1, 15))

    elements.append(Paragraph("2. Risk Assessment", section_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 8))

    risk_percent = prob * 100

    if prob < 0.30:
        risk_level = "Low Risk"
        color = "green"
    elif prob < 0.70:
        risk_level = "Moderate Risk"
        color = "orange"
    else:
        risk_level = "High Risk"
        color = "red"

    elements.append(Paragraph(
        f"<b>Predicted Risk Score:</b>",
        normal_style
    ))
    
    elements.append(Spacer(1, 5))
    
    elements.append(Paragraph(
        f"<font color='{color}' size=14><b>{risk_percent:.2f}%</b></font>",
        normal_style
    ))

    elements.append(Spacer(1,8))
    elements.append(Paragraph(
        f"<b>Risk Category:</b> <font color='{color}'><b>{risk_level}</b></font>",
        normal_style
    ))

    risk_chart = create_risk_chart(prob)
    elements.append(Spacer(1, 10))
    elements.append(Image(risk_chart, width=400, height=200))

    elements.append(Spacer(1, 15))

    elements.append(Paragraph("3. Clinical Interpretation", section_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 8))

    explanation = generate_clinical_explanation(user_input, prob, disease)
    elements.append(Paragraph(explanation, normal_style))

    elements.append(Spacer(1, 15))

    elements.append(Paragraph("4. Key Risk Drivers", section_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 8))
    
    elements.append(Paragraph(
        "Top features influencing the model prediction:",
        normal_style
    ))
    
    if disease == "diabetes":
        pipeline = diabetes_pipeline
        feature_names = [
            "gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"
        ]
    else:
        pipeline = heart_pipeline
        feature_names = [
            "age", "gender", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak",
            "slope", "ca", "thal"
        ]
        
    fi_chart = create_feature_importance_chart(pipeline, feature_names)
    
    elements.append(Spacer(1, 10))
    elements.append(Image(fi_chart, width=400, height=200))

    elements.append(Spacer(1, 15))

    elements.append(Paragraph("5. Recommendations", section_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 8))

    if disease == "diabetes":
        suggestions, _ = generate_counterfactuals(user_input, diabetes_pipeline, disease)
    else:
        suggestions, _ = generate_counterfactuals(user_input, heart_pipeline, disease)

    if suggestions:
        
        best = suggestions[0]
        
        elements.append(Paragraph(
            f"<b>Most Impactful Change:</b><br/>"
            f"{best['label']}: {best['current']} → {best['optimal']}<br/>"
            f"Estimated Risk Reduction: <b>{best['improvement']*100:.2f}%</b>",
            normal_style
        ))
        
        elements.append(Spacer(1, 10))
        
        if len(suggestions) > 1:
            elements.append(Paragraph("<b>Other Recommendations:</b>", normal_style))
            elements.append(Spacer(1, 5))
            
            for s in suggestions[1:]:
                elements.append(Paragraph(
                    f"- {s['label']} ({s['current']} → {s['optimal']}) "
                    f"(↓ {s['improvement']*100:.2f}%)",
                    normal_style
                ))
    
    else:
        elements.append(Paragraph(
            "Maintain a healthy lifestyle and consult a healthcare professional.",
            normal_style
        ))

    elements.append(Spacer(1, 15))

    elements.append(Paragraph("6. Disclaimer", section_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph(
        "This report is generated using a machine learning model and is intended for informational purposes only. "
        "It should not be considered a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare provider.",
        normal_style
    ))

    elements.append(Spacer(1, 20))

    elements.append(Paragraph(
        "Medical Report Generated by ML-Based Clinical System",
        footer_style
    ))

    doc.build(elements)
    
    return file_name

st.title("Disease Risk Prediction & Analysis System")

tab1, tab2, tab3 = st.tabs([
    "Diabetes Risk",
    "Heart Disease Risk",
    "Clinical Report"
])


with tab1:
    st.header("Diabetes Risk Assessment")

    st.subheader("Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
         age = st.number_input("Age", 0, 120)     

    st.subheader("Medical History")
    col1, col2 = st.columns(2)

    with col1:
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        hypertension_val = 1 if hypertension == "Yes" else 0
        smoking = st.selectbox(
            "Smoking History",
            ["never", "former", "current", "not current", "ever", "No Info"]
        )
    
    with col2:
        heart_disease = st.selectbox("Heart Disease History", ["No", "Yes"])
        heart_disease_val = 1 if heart_disease == "Yes" else 0

    st.subheader("Clinical Measurements")
    col1, col2 = st.columns(2)

    with col1:
        bmi = st.number_input("BMI", 0.0)

    with col2:
        hba1c = st.number_input("HbA1c Level", 0.0)
        
    glucose = st.number_input("Blood Glucose Level", 0.0)

    if st.button("Analyze Diabetes Risk"):
        
        if age <= 0:
            st.error("Age must be greater than 0")
            st.stop()
            
        if bmi <= 0:
            st.error("BMI must be greater than 0")
            st.stop()
            
        if hba1c <= 0:
            st.error("HbA1c must be greater than 0")
            st.stop()
            
        if glucose <= 0:
            st.error("Blood Glucose must be greater than 0")
            st.stop()

        input_df = pd.DataFrame([{
            'gender': gender,
            'age': age,
            'hypertension': hypertension_val,
            'heart_disease': heart_disease_val,
            'smoking_history': smoking,
            'bmi': bmi,
            'HbA1c_level': hba1c,
            'blood_glucose_level': glucose
        }])
        
        raw_prob = diabetes_pipeline.predict_proba(input_df)[0][1]
        
        adjusted_prob, reasons = apply_clinical_override(
            input_df.iloc[0].to_dict(),
            raw_prob,
            "diabetes"
        )
                
        st.session_state.diabetes_result = {
            "prob": adjusted_prob,
            "raw_prob": raw_prob,
            "override_reason": reasons,
            "input": input_df.iloc[0].to_dict()
        }
        
        show_result(adjusted_prob)
        
        if reasons:
            st.warning(f"Clinical Override Applied: {reasons}")

        with st.expander("Key Risk Drivers"):
            show_feature_importance(diabetes_pipeline)
        
        with st.expander("Clinical Findings"):
            show_clinical_report(
                st.session_state.diabetes_result["input"],
                "diabetes"
            )

        with st.expander("Clinical Interpretation"):
            show_clinical_explanation(
                st.session_state.diabetes_result["input"],
                adjusted_prob,
                "diabetes"
            )

        with st.expander("Risk Reduction Suggestions"):
            show_counterfactuals(
                diabetes_pipeline,
                st.session_state.diabetes_result["input"],
                "diabetes"
            )

    st.markdown("""
    <div style="
        background-color: #f1f3f6;
        padding: 10px;
        border-radius: 8px;
        font-size: 13px;
        color: #555;">
        "This is an ML-based risk estimate and not a medical diagnosis.
        Consult a healthcare professional for clinical decisions."
    </div>
    """, unsafe_allow_html=True)
        
with tab2:
    st.header("Heart Disease Risk Assessment")

    st.subheader("Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 20, 100, key="h_age")

    with col2:
        gender = st.selectbox("Gender", ["Female", "Male"], key="h_gender")

    st.subheader("Clinical Symptoms")
    col1, col2 = st.columns(2)

    with col1:
        cp_label = st.selectbox("Chest Pain Type", list(cp_map.keys()))
        cp = cp_map[cp_label]

    with col2:
        exang_label = st.selectbox("Exercise Angina", list(exang_map.keys()))
        exang = exang_map[exang_label]

    st.subheader("Medical Measurements")
    col1 , col2 = st.columns(2)

    with col1:
        trestbps = st.number_input("Resting Blood Pressure", 80, 200)
        chol = st.number_input("Cholesterol", 100, 600)
        fbs_label = st.selectbox("Fasting Blood Sugar", list(fbs_map.keys()))
        fbs = fbs_map[fbs_label]

    with col2:
        restecg_label = st.selectbox("Rest ECG", list(restecg_map.keys()))
        restecg = restecg_map[restecg_label]
        thalach = st.number_input("Max Heart Rate", 60, 220)
        oldpeak = st.number_input("ST Depression", 0.0, 6.0)

    st.subheader("Advanced Parameters")
    col1, col2 = st.columns(2)

    with col1:
        slope_label = st.selectbox("Slope", list(slope_map.keys()))
        slope = slope_map[slope_label]

    with col2:
        ca_label = st.selectbox("Major Vessels", list(ca_map.keys()))
        ca = ca_map[ca_label]

    thal_label = st.selectbox("Thalassemia", list(thal_map.keys()))
    thal = thal_map[thal_label]

    if st.button("Analyze Heart Risk"):
        
        if age <= 0:
            st.error("Age must be valid")
            st.stop()
            
        if trestbps <= 0:
            st.error("Blood Pressure must be valid")
            st.stop()
            
        if chol <= 0:
            st.error("Cholesterol must be valid")
            st.stop()
            
        if thalach <= 0:
            st.error("Max Heart Rate must be valid")
            st.stop()

        input_df = pd.DataFrame([{
            'age': age,
            'gender': 1 if gender == "Male" else 0,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }])

        raw_prob = heart_pipeline.predict_proba(input_df)[0][1]
        
        adjusted_prob, reasons = apply_clinical_override(
            input_df.iloc[0].to_dict(),
            raw_prob,
            "heart"
        )

        st.session_state.heart_result = {
            "prob": adjusted_prob,
            "raw_prob": raw_prob,
            "override_reason": reasons,
            "input": input_df.iloc[0].to_dict()
        }
        
        show_result(adjusted_prob)
        
        if reasons:
            st.warning(f"Clinical Override Applied: {reasons}")

        with st.expander("Key Risk Drivers"):
            show_feature_importance(heart_pipeline)
        
        with st.expander("Clinical Findings"):
            show_clinical_report(
                st.session_state.heart_result["input"],
                "heart"
            )

        with st.expander("Clinical Interpretation"):
            show_clinical_explanation(
                st.session_state.heart_result["input"],
                adjusted_prob,
                "heart"
            )

        with st.expander("Risk Reduction Suggestions"):
            show_counterfactuals(
                heart_pipeline,
                st.session_state.heart_result["input"],
                "heart"
            )

    st.markdown("""
    <div style="
        background-color: #f1f3f6;
        padding: 10px;
        border-radius: 8px;
        font-size: 13px;
        color: #555;">
        "This is an ML-based risk estimate and not a medical diagnosis.
        Consult a healthcare professional for clinical decisions."
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.header(" Clinical Report Generator")

    report = "Clinical Report\n"
    report += f"Generated on: {datetime.now()}\n\n"
    
    if st.session_state.diabetes_result is not None:
        if st.button("Generate Diabetes Report"):
            file_path = generate_pdf_report(
                st.session_state.diabetes_result["input"],
                st.session_state.diabetes_result["prob"],
                "diabetes"
            )
            st.success("Report generated successfully!")

            st.session_state.diabetes_pdf = file_path

        if "diabetes_pdf" in st.session_state:
            with open(st.session_state.diabetes_pdf, "rb") as f:
                st.download_button(
                    label="Download Diabetes PDF",
                    data=f,
                    file_name="diabetes_report.pdf",
                    mime="application/pdf"
                )
                
    if st.session_state.heart_result is not None:
        if st.button("Generate Heart Report"):
            file_path = generate_pdf_report(
                st.session_state.heart_result["input"],
                st.session_state.heart_result["prob"],
                "heart"
            )
            st.success("Report generated successfully!")
            
            st.session_state.heart_pdf = file_path

        if "heart_pdf" in st.session_state:
            with open(st.session_state.heart_pdf, "rb") as f:
                st.download_button(
                    label="Download Heart PDF",
                    data=f,
                    file_name="heart_report.pdf",
                    mime="application/pdf"
                )

    st.warning("This report is generated by a machine learning model and is not a substitute for professional medical advice.")
