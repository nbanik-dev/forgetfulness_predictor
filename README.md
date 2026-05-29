# 🧠 Forgetfulness Predictor Using Study Patterns

A Machine Learning project that predicts **how much information a student still remembers** after studying — based on their study habits. The idea is inspired by the famous **Ebbinghaus Forgetting Curve**, which says that we forget information over time unless we revise it.

---

## 📌 What This Project Does

You enter a few details about your study, such as:

- How many days ago you studied
- How many times you revised
- How long you studied (in hours)
- How difficult the topic is
- Your CGPA

…and the system predicts your **retention percentage** — how much you likely still remember.

It also gives you a suggestion:

| Retention | Suggestion |
|-----------|------------|
| Low (below 40%) | ⚠️ Revise Immediately |
| Medium (40–70%) | 🕐 Revise Soon |
| High (above 70%) | ✅ Retention is Good |

---

## 💡 The Idea Behind It

The **Ebbinghaus Forgetting Curve** is a well-known theory in psychology. It explains that human memory fades over time if we don't review what we learned. But every time we revise, we remember it for longer.

This project turns that theory into a real, working prediction tool using Machine Learning.

---

## 🛠️ Technologies Used

- **Python** – the main programming language
- **Pandas** – to handle the dataset
- **Scikit-learn** – for the Machine Learning models
- **Matplotlib** – to draw graphs
- **Streamlit** – to create the web app interface

---

## 🤖 How the Machine Learning Works

1. A dataset of 1000 students' study patterns is created (`data.csv`).
2. The data is split into **80% training** and **20% testing**.
3. Two models are trained and compared:
   - **Linear Regression** (simple model)
   - **Random Forest** (advanced model)
4. The model with the **lower error (RMSE)** is chosen as the final model.

**Result:** Random Forest performed better.

| Model | RMSE (lower is better) |
|-------|------------------------|
| Linear Regression | 10.53 |
| Random Forest ✅ | 5.80 |

---

## 📂 Project Files

| File | What it does |
|------|--------------|
| `app.py` | The main app (interface + ML prediction) |
| `generate_data.py` | Creates the dataset (`data.csv`) |
| `data.csv` | The dataset used to train the model |
| `requirements.txt` | List of libraries needed to run the project |

---

## ▶️ How to Run This Project

### Step 1 — Install Python
Download and install Python from [python.org](https://www.python.org/downloads).
✅ During installation, check **"Add Python to PATH"**.

### Step 2 — Install the required libraries
Open a terminal inside the project folder and run:
```bash
pip install -r requirements.txt
```

### Step 3 — Run the app
```bash
python -m streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501` with the app running. 🎉

---

## 📊 Features of the App

The app has **3 tabs**:

1. **🔮 Predict** – Enter your study details and get your retention prediction with a suggestion.
2. **📊 Model Performance** – See how well the models perform, with charts comparing actual vs predicted values.
3. **📈 Forgetting Curve** – An interactive graph showing how your memory fades over 30 days, and how revising slows down that forgetting.

---

## 🚀 Future Improvements

- Collect **real student data** through surveys instead of synthetic data
- Build a **mobile app** version
- Add **AI-based study recommendations**
- Turn it into a complete **smart academic assistant**

---

## 👤 Author

Created as a Machine Learning academic project.

---

*This project demonstrates how a classic psychology theory can be combined with Machine Learning to build a helpful, personalized learning tool.*
