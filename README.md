Demo Video Link: https://1drv.ms/f/c/fa948cb5c589cd6e/IgAlbKbY9AhfR4zYAhH2SEZbAdcszE-QQ4ruzqm1Aclwxc0?e=4UOb4c

TRAILERFLOW
AI-Powered Manufacturing Bay Scheduler

Overview
TrailerFlow is a production scheduling prototype that dynamically prioritizes units into four active bays and a queue. It combines AI-based email delay extraction with predictive modeling to automatically adjust manufacturing schedules.

The system demonstrates how unstructured operational updates (e.g., Outlook emails) can be converted into structured delay inputs and used to intelligently reorder production bays.

---

Problem

In manufacturing environments:

* Scheduling updates frequently arrive via email.
* Units are large, heavy, and costly to physically reshuffle.
* Reported delays often understate real operational impact.
* Manual schedule adjustments are reactive and inefficient.

TrailerFlow addresses this by automatically recalculating schedule priority when delays are introduced.

---

Core Logic

1. Four Active Bays
   The fastest predicted unit goes to Bay 1.
   The longest predicted unit goes to Bay 4.
   Remaining units are placed in the Queue.

2. Delay Modeling
   Each unit can receive structured delays:

   * parts_days
   * accident_days
   * training_days

3. Predictive Adjustment
   Predicted completion time =
   estimated_days + delay inputs + historical adjustment

The model accounts for discrepancies beyond reported delays based on training data.

---

AI Components

1. Google Gemini API
   Extracts structured delay information from email-style text.

2. Linear Regression Model
   Trained on historical dataset samples to predict completion time.

Model experimentation was performed in Jupyter Notebook.
The dashboard is built using Streamlit in VS Code.

---

Dashboard Features

* Kanban-style layout representing 4 production bays and a queue
* Automatic re-ranking after delay updates
* Bay movement notifications
* Operational metrics (Top 4 bays only):

  * Bay Overrun Exposure
  * Bay Movements
  * Queue Backlog

---

Tech Stack

Python
Streamlit
Pandas
Scikit-learn
Google Gemini API
Jupyter Notebook
VS Code

---

Setup (Windows)

1. Clone Repository

git clone (https://github.com/Ramani-SV/hackathon.git)

2. Create Virtual Environment

python -m venv .venv
.venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

If no requirements file exists:

pip install streamlit pandas python-dotenv google-generativeai scikit-learn

4. Add Google API Key

Create a file named:

.env

Add:

GEMINI_API_KEY=your_google_gemini_api_key_here

Important:

* Do not commit the .env file to GitHub.
* Add .env to .gitignore.

If you do not have a Gemini API key, request one from Google AI Studio or contact the repository owner for testing access. Free-tier keys may encounter rate limits.

5. Run the Application

streamlit run app.py

Open the local URL shown in the terminal (typically [http://localhost:8501](http://localhost:8501)).


