from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import google.generativeai as genai
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import ConnectionFailure
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import logging
import motor.motor_asyncio





app = FastAPI()




# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Kết nối đến MongoDB
uri = "mongodb+srv://hoangkhadooo:2tPvSPwcTi29VhAF@cluster0.dz6kf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = AsyncIOMotorClient(uri)

# Chọn cơ sở dữ liệu
db = client.get_database("hoangkha")

# Send a ping to confirm a successful connection
try:
    client = AsyncIOMotorClient(uri)
    client.admin.command('ping')  # Gửi ping để kiểm tra kết nối
    print("Kết nối đến MongoDB thành công!")
except ConnectionFailure as e:
    print(f"Kết nối thất bại: {e}")
# Cấu hình Gemini API
genai.configure(api_key="AIzaSyBkNYQEPaBMOJz6JzYnU--oI8JY5EXkJKk")

# Tải mô hình và scaler cho từng endpoint
model_semester_change = joblib.load('./model/best_model.pkl')
scaler_semester_change = joblib.load('./model/label.pkl')
dcm = joblib.load('./model/model.joblib')
encoders = joblib.load('./model/encoders.joblib')
model = genai.GenerativeModel("gemini-1.5-flash")


# User model
class User(BaseModel):
    email: str
    password: str

# Profile model
class Profile(BaseModel):
    studentID: str
    full_name: str
    email: str

# Endpoint to sign up a user
@app.post("/signup")
async def signup(user: User):
    try:
        # In thông tin người dùng ra console
        print(f"Received user info: email={user.email}, password={user.password}")

        # Kiểm tra xem email đã tồn tại chưa
        existing_user = await db.users.find_one({"email": user.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="Email đã tồn tại.")

        # Tạo người dùng mới trong MongoDB
        user_data = {
            'email': user.email,
            'password': user.password  # Ghi nhớ: Nên mã hóa mật khẩu
        }
        result = await db.users.insert_one(user_data)

        # In thông tin người dùng mới đã được lưu
        print(f"User registered with uid: {result.inserted_id}")

        return {"message": "Đăng ký thành công!", "uid": str(result.inserted_id)}

    except Exception as e:
        print(f"Error during signup: {str(e)}")  # In lỗi ra console
        raise HTTPException(status_code=400, detail=f"Đăng ký thất bại: {str(e)}")


# Endpoint to log in a user
@app.post("/login")
async def login(user: User):
    try:
        user_record = await db.users.find_one({"email": user.email})
        if user_record is None or user_record['password'] != user.password:
            raise HTTPException(status_code=401, detail="Email hoặc mật khẩu không đúng.")

        return {"message": "Đăng nhập thành công!", "uid": str(user_record['_id'])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to save user profile
@app.post("/profile")
async def save_profile(profile: Profile):
    try:
        user_data = {
            'studentID': profile.studentID,
            'full_name': profile.full_name,
            'email': profile.email
        }
        await db.users.update_one({'email': profile.email}, {'$set': user_data}, upsert=True)
        return {"message": "Hồ sơ đã được lưu thành công!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lưu hồ sơ thất bại: {str(e)}")

# Endpoint to get user information
@app.get("/user/{email}")
async def get_user_info(email: str):
    try:
        user_record = await db.users.find_one({"email": email})
        if user_record is None:
            raise HTTPException(status_code=404, detail="Người dùng không tồn tại")

        return {
            "full_name": user_record.get("full_name", None),
            "studentID": user_record.get("studentID", None)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    




# Define your request model
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
    print("Received data:", data)
    input_data = pd.DataFrame([data.dict()])
    print("Input DataFrame:", input_data)

    # Đổi tên cột để phù hợp với dữ liệu huấn luyện
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
    print("Renamed columns:", input_data.columns)

    # Kiểm tra xem tất cả các cột yêu cầu có mặt hay không
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

    # Mã hóa dữ liệu
    for col in input_data.columns:
        if col in encoders:  # Giả sử bạn đã định nghĩa các bộ mã hóa trước đó
            input_data[col] = encoders[col].transform(input_data[col])
            
    print("Processed input data:", input_data)

    # Dự đoán
    try:
        prediction = dcm.predict(input_data)  # dcm là mô hình của bạn
        print("Prediction result:", prediction)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during prediction.")

    # Tạo prompt cho GPT
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
        # Khởi tạo và gọi mô hình Gemini
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
        print("GPT output:", gpt_output)

        

        return {
            "prediction": prediction[0],
            "GPT_Recommendation": gpt_output
        }
    except Exception as e:
        print(f"Error in GPT diagnosis: {str(e)}")
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
predictions_collection = db['predictions']
class StudentData(BaseModel):
    StudentID: str
    Gender: str
    Year_of_Study: str
    Attendance: float
    Academic_Score: float
    Extracurricular_Activities: str
    Social_Interactions: str
    Behavior_Issues: str
    Peer_Relationships: str
    Classroom_Behavior: str
    Email: str  # Thêm trường email

async def get_latest_record_by_email(email: str):
    records_ref = db.predictions  # Truy cập vào bộ sưu tập "predictions"
    
    # Truy vấn bất đồng bộ
    latest_record = await records_ref.find_one(
        {"email": email},
        sort=[("timestamp", -1)]  # Sắp xếp theo timestamp giảm dần
    )
    
    return latest_record

@app.post("/predict-psychology")
async def predict_psychology(student_data: StudentData):
    latest_record = await get_latest_record_by_email(student_data.Email)  # Sử dụng await

    if not latest_record:
        return {"error": "Không tìm thấy bản ghi nào cho email này."}

    prediction = latest_record.get("prediction")
    recommendation = latest_record.get("recommendation")
    timestamp = latest_record.get("timestamp")

    # Tạo prompt để gửi đến mô hình
    prompt = (
        f"Dựa trên dữ liệu học sinh sau: \n"
        f"Student ID: {student_data.StudentID}\n"
        f"Gender: {student_data.Gender}\n"
        f"Year of Study: {student_data.Year_of_Study}\n"
        f"Attendance: {student_data.Attendance}%\n"
        f"Academic Score: {student_data.Academic_Score}\n"
        f"Extracurricular Activities: {student_data.Extracurricular_Activities}\n"
        f"Social Interactions: {student_data.Social_Interactions}\n"
        f"Behavior Issues: {student_data.Behavior_Issues}\n"
        f"Peer Relationships: {student_data.Peer_Relationships}\n"
        f"Classroom Behavior: {student_data.Classroom_Behavior}\n"
        f"Dự đoán tâm lý: {prediction}\n"
        f"Khuyến nghị: {recommendation}\n"
        f"Timestamp: {timestamp}\n"
        f"Hãy tạo một phân tích tâm lý cho học sinh này và cho tôi kết quả cuối cùng về sức khỏe tâm thần của học sinh này (output là tiếng việt)."
    )

    # Gọi API Gemini để tạo phản hồi
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=2000,  # Bạn có thể điều chỉnh số token tối đa
            temperature=0.1,  # Bạn có thể điều chỉnh độ ngẫu nhiên
        ),
    )
    geminioutput = response.text.strip()
    
    return {"response": geminioutput}

class PredictionData(BaseModel):
    email: str
    prediction: str
    GPT_Recommendation: str
    timestamp: datetime

@app.post("/save_prediction")
async def save_prediction(data: PredictionData):
    try:
        predictions_ref = db["predictions"]  # Sử dụng db["collection_name"] để kết nối
        logging.info("Connected to MongoDB predictions collection")
        
        # Kiểm tra dữ liệu trước khi lưu
        logging.info("Data to be saved: %s", data.dict())

        # Thử lưu dữ liệu vào MongoDB
        result = await predictions_ref.insert_one(data.dict())
        logging.info(f"Inserted document ID: {result.inserted_id}")

        return {"message": "Prediction saved successfully"}
    
    except Exception as e:
        logging.error("Error saving prediction: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error saving prediction: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
