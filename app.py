from fastapi import FastAPI, UploadFile, File
import pdfplumber
import pytesseract
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sqlite3
import shutil
import os

# Set tesseract path (required for deployment)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI()

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("contracts.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS contracts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  text TEXT,
                  summary TEXT,
                  risks TEXT)''')
    conn.commit()
    conn.close()

def insert_contract(name, text, summary, risks):
    conn = sqlite3.connect("contracts.db")
    c = conn.cursor()
    c.execute("INSERT INTO contracts (name, text, summary, risks) VALUES (?, ?, ?, ?)",
              (name, text, summary, risks))
    conn.commit()
    conn.close()

init_db()

# ---------------- MODEL ----------------
# Use smaller model (important for free deployment)
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

# ---------------- LOGIC ----------------
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_image(file_path):
    img = Image.open(file_path)
    return pytesseract.image_to_string(img)

def summarize_contract(text):
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(
        inputs['input_ids'],
        num_beams=4,
        max_length=150,
        min_length=40,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def analyze_risks(text):
    risks = []
    lower_text = text.lower()

    if "penalty" in lower_text:
        risks.append("Contains penalty clause")
    if "auto-renew" in lower_text:
        risks.append("Contains auto-renewal clause")
    if "termination" not in lower_text:
        risks.append("No clear termination clause")
    if "liable for all damages without limitation" in lower_text:
        risks.append("Uncapped liability")

    return risks if risks else ["No major risks detected"]

# ---------------- API ----------------
@app.get("/")
def home():
    return {"message": "AI Contract Reviewer API Running"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    else:
        text = extract_text_from_image(file_path)

    summary = summarize_contract(text)
    risks = analyze_risks(text)

    insert_contract(file.filename, text, summary, ", ".join(risks))

    os.remove(file_path)

    return {
        "summary": summary,
        "risks": risks
    }