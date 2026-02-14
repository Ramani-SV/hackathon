# TrailerFlow Dashboard

TrailerFlow is a modern dashboard for managing trailer repair schedules using Machine Learning and AI. It predicts completion days using Linear Regression and extracts delay updates from emails using the Gemini AI API.

## Features

- **ML Predictions**: Uses Linear Regression to predict `actual_days` based on trailer type and various delay factors.
- **Kanban Scheduler**: Automatically assigns trailers to 4 repair bays or a queue based on their predicted completion timeline.
- **AI Email Extraction**: Paste an email update, and the app uses Gemini AI to extract trailer IDs and delay changes (parts, accidents, training).
- **Dynamic Updates**: Adding new trailers or updating existing ones triggers an immediate recalculation of the schedule and bay placement.

## Project Structure

- `app.py`: The main Streamlit dashboard UI.
- `model.py`: ML logic for training the model and making predictions.
- `gemini_extractor.py`: Logic for interacting with the Gemini API.
- `trailerflow_dataset_100.csv`: The training dataset.
- `.env.example`: Template for environment variables.
- `requirements.txt`: Project dependencies.

## Setup Instructions

1. **Clone or Extract** the project files.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure API Key**:
   - Create a file named `.env` in the root directory.
   - Copy the content from `.env.example` into `.env`.
   - Replace `your_gemini_api_key_here` with your actual Gemini API key.
4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## Usage

- **Top Metrics**: Overview of fleet status and risk levels.
- **Kanban Bays**: Visual representation of repair priority.
- **Add New Trailer**: Enter details to add a trailer to the fleet.
- **Email Update**: Paste text like *"Trailer T103 is delayed by 3 days due to missing parts"* and click "Extract Delay Using AI" to update the schedule automatically.
