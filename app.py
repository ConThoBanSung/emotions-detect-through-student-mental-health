from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import google.generativeai as genai

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình Gemini API
genai.configure(api_key="AIzaSyBkNYQEPaBMOJz6JzYnU--oI8JY5EXkJKk")

# Tải mô hình và scaler cho từng endpoint
model_semester_change = joblib.load('best_model.pkl')
scaler_semester_change = joblib.load('label.pkl')
model = joblib.load('model.joblib')
encoders = joblib.load('encoders.joblib')

# Định nghĩa mô hình dữ liệu cho từng endpoint

# Endpoint 1: predict_semester_change
class SemesterChangeRequest(BaseModel):
    Gender: int
    Attendance_: float
    Academic_Score: float
    Classroom_Behavior: float
    Stress_Level: float
    Anxiety_Level: float
    Sleep_Quality: float
    Mental_Health_Score: float
    Social_Engagement: float
    Semester: int

@app.post("/predict_semester_change")
async def predict_semester_change(data: List[SemesterChangeRequest]):
    df = pd.DataFrame([d.dict() for d in data])
    
    # Đổi tên cột để khớp với mô hình
    df.columns = ['Gender', 'Attendance (%)', 'Academic Score', 'Classroom Behavior', 
                  'Stress Level', 'Anxiety Level', 'Sleep Quality', 'Mental Health Score', 
                  'Social Engagement', 'Semester']
    
    # Tiền xử lý dữ liệu
    df_scaled = scaler_semester_change.transform(df)
    
    # Dự đoán nguy cơ bỏ học
    try:
        predictions = model_semester_change.predict(df_scaled)
        df['Risk of Dropout'] = predictions
        
        # Phân tích sự thay đổi học kỳ
        result = df.groupby('Semester')['Risk of Dropout'].mean().to_dict()
        
        # Tạo prompt cho GPT
        prompt = (
            "Tôi đang có tâm trạng và nguy cơ bỏ học.\n"
            "Đánh giá tâm lí của tôi và giúp tôi nếu tâm lý tôi đang không ổn định và có nguy cơ bỏ học (output vừa vừa thôi).\n"
            f"Thông tin: {df.to_dict(orient='records')}\n"
        )
        
        # Khởi tạo và gọi Gemini Model
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=2000,
                temperature=0.1,
            ),
        )
        
        gpt_output = response.text.strip()
        
        return {
            "Semester Change Analysis": result,
            "GPT Recommendation": gpt_output
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint 2: predict
class MentalHealthRequest(BaseModel):
    How_often_do_you_feel_overwhelmed_by_your_studies: str
    Do_you_have_difficulty_sleeping_due_to_stress: str
    How_satisfied_are_you_with_your_current_academic_performance: str
    How_often_do_you_engage_in_social_activities_outside_of_school: str
    How_frequently_do_you_feel_anxious_about_upcoming_exams: str
    How_would_you_rate_your_overall_mental_well_being: str
    Do_you_find_it_hard_to_concentrate_during_lectures: str
    How_often_do_you_feel_pressure_from_your_family_regarding_your_studies: str
    How_often_do_you_feel_that_your_workload_is_manageable: str
    Do_you_feel_that_you_have_sufficient_support_from_friends_and_family: str

@app.post("/predict")
def predict(data: MentalHealthRequest):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Rename columns to match training data
    input_data.columns = [
        'How often do you feel overwhelmed by your studies?',
        'Do you have difficulty sleeping due to stress?',
        'How satisfied are you with your current academic performance?',
        'How often do you engage in social activities outside of school?',
        'How frequently do you feel anxious about upcoming exams?',
        'How would you rate your overall mental well-being?',
        'Do you find it hard to concentrate during lectures?',
        'How often do you feel pressure from your family regarding your studies?',
        'How often do you feel that your workload is manageable?',
        'Do you feel that you have sufficient support from friends and family?'
    ]
    
    # Check if all required columns are present
    required_columns = [
        'How often do you feel overwhelmed by your studies?',
        'Do you have difficulty sleeping due to stress?',
        'How satisfied are you with your current academic performance?',
        'How often do you engage in social activities outside of school?',
        'How frequently do you feel anxious about upcoming exams?',
        'How would you rate your overall mental well-being?',
        'Do you find it hard to concentrate during lectures?',
        'How often do you feel pressure from your family regarding your studies?',
        'How often do you feel that your workload is manageable?',
        'Do you feel that you have sufficient support from friends and family?'
    ]
    
    for col in required_columns:
        if col not in input_data.columns:
            raise HTTPException(status_code=400, detail=f"Missing field: {col}")

    # Encode data
    for col in input_data.columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Tạo prompt cho GPT với dữ liệu từ frontend
    prompt = (
        f"Những thông tin sau đây được cung cấp bởi tôi:\n"
        f"How often do you feel overwhelmed by your studies? {data.How_often_do_you_feel_overwhelmed_by_your_studies}\n"
        f"Do you have difficulty sleeping due to stress? {data.Do_you_have_difficulty_sleeping_due_to_stress}\n"
        f"How satisfied are you with your current academic performance? {data.How_satisfied_are_you_with_your_current_academic_performance}\n"
        f"How often do you engage in social activities outside of school? {data.How_often_do_you_engage_in_social_activities_outside_of_school}\n"
        f"How frequently do you feel anxious about upcoming exams? {data.How_frequently_do_you_feel_anxious_about_upcoming_exams}\n"
        f"How would you rate your overall mental well-being? {data.How_would_you_rate_your_overall_mental_well_being}\n"
        f"Do you find it hard to concentrate during lectures? {data.Do_you_find_it_hard_to_concentrate_during_lectures}\n"
        f"How often do you feel pressure from your family regarding your studies? {data.How_often_do_you_feel_pressure_from_your_family_regarding_your_studies}\n"
        f"How often do you feel that your workload is manageable? {data.How_often_do_you_feel_that_your_workload_is_manageable}\n"
        f"Do you feel that you have sufficient support from friends and family? {data.Do_you_feel_that_you_have_sufficient_support_from_friends_and_family}\n"
        "Dựa vào các thông tin trên, hãy đưa ra đánh giá và lời khuyên về sức khỏe tâm lý của tôi.\n"
    )

    try:
        # Khởi tạo và gọi Gemini Model
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=2000,
                temperature=0.1,
            ),
        )
        
        gpt_output = response.text.strip()
        
        return {
            "prediction": prediction[0],
            "GPT_Recommendation": gpt_output
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in GPT diagnosis: {str(e)}")
    
class GPTRequest(BaseModel):
    dropout_risk: int
    emotion: str

@app.post("/gpt_diagnosis")
async def gpt_diagnosis(gpt_request: GPTRequest):
    try:
        prompt = (
            f"Học sinh đang có tâm trạng '{gpt_request.emotion}' và nguy cơ bỏ học {gpt_request.dropout_risk}. ( trong đó dropout_risk với 1 là có nguy cơ bỏ học và 0 là không có nguy cơ bỏ học )\n"
            "đánh giá học sinh và giúp học sinh nếu tâm lí học sinh đang không ổn định và có nguy cơ bỏ học (output là tiếng Việt) \n"
        )

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=2000,  
                temperature=0.1,
            ),
        )

        gpt_output = response.text.strip()

        return {"recommendation": gpt_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in GPT diagnosis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
