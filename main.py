from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")


model = joblib.load("xgboost_model.pkl")

gender_map = {"Male": 1, "Female": 0}
married_map = {"Yes": 1, "No": 0}
work_type_map = {"Private": 0,"Self-employed": 1,"Govt_job": 2,"Children": 3,"Never_worked": 4}
residence_map = {"Urban": 1, "Rural": 0}
smoking_status_map = {"formerly smoked": 0,"never smoked": 1,"smokes": 2,"Unknown": 3}

AGE_MIN, AGE_MAX = 0.08, 82.0
GLUCOSE_MIN, GLUCOSE_MAX = 55.12, 271.74
BMI_MIN, BMI_MAX = 10.3, 97.6

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def categorize_age(age):
    if age < 40:
        return 0  
    elif age < 60:
        return 1  
    else:
        return 2  

def categorize_bmi(bmi):
    if bmi < 18.5:
        return 0  
    elif bmi < 25:
        return 1  
    elif bmi < 30:
        return 2  
    else:
        return 3  

def categorize_glucose(glucose):
    if glucose < 70:
        return 0  
    elif glucose < 100:
        return 1  
    else:
        return 2  

@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    gender: str = Form(...),
    age: float = Form(...),
    hypertension: int = Form(...),
    heart_disease: int = Form(...),
    ever_married: str = Form(...),
    work_type: str = Form(...),
    Residence_type: str = Form(...),
    avg_glucose_level: float = Form(...),
    bmi: float = Form(...),
    smoking_status: str = Form(...)
):
    try:
        gender_mapped = gender_map[gender]
        married_mapped = married_map[ever_married]
        work_type_mapped = work_type_map[work_type]
        residence_mapped = residence_map[Residence_type]
        smoking_status_mapped = smoking_status_map[smoking_status]

        age_norm = normalize(age, AGE_MIN, AGE_MAX)
        glucose_norm = normalize(avg_glucose_level, GLUCOSE_MIN, GLUCOSE_MAX)
        bmi_norm = normalize(bmi, BMI_MIN, BMI_MAX)

        age_category = categorize_age(age)
        bmi_category = categorize_bmi(bmi)
        glucose_category = categorize_glucose(avg_glucose_level)

        input_array = np.array([
            gender_mapped,
            age_norm,
            hypertension,
            heart_disease,
            married_mapped,
            work_type_mapped,
            residence_mapped,
            glucose_norm,
            bmi_norm,
            smoking_status_mapped,
            age_category,
            bmi_category,
            glucose_category
        ]).reshape(1, -1)

        print("Input Array Shape:", input_array.shape)

        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1]  

        result = "🧠 High Risk of Stroke" if prediction == 1 else "✅ Low Risk"
        risk_percent = round(probability * 100, 2)

        return templates.TemplateResponse("result.html", {"request": request,"result": result,"probability": risk_percent})
    
    except KeyError as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": f"Invalid category selected: {e}"})
    except ValueError as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": f"Invalid input value: {e}"})
    except Exception as e:
        return templates.TemplateResponse("error.html", {"request": request, "error": f"An unexpected error occurred: {str(e)}"})