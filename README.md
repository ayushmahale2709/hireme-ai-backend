# HireMe AI Backend

FastAPI backend for **HireMe AI**, a resume-powered AI interview assistant.

The backend parses Ayush Mahale's resume into structured information and uses the **Groq API** with `openai/gpt-oss-120b` to answer recruiter questions using only verified resume information.

## Features

* Resume PDF extraction using PyPDF2
* Structured resume parsing with Pydantic
* AI-powered recruiter Q&A
* Streaming AI responses
* Conversation context for follow-up questions
* FastAPI REST API
* CORS support for frontend integration
* Groq API integration

## Tech Stack

* Python
* FastAPI
* Uvicorn
* Groq API
* GPT-OSS 120B
* Pydantic
* PyPDF2
* Python-dotenv

## Project Structure

```text
backend/
├── main.py
├── Ayush_Mahale_Resume.pdf
├── requirements.txt
├── .env
└── README.md
```

## Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/hireme-ai-backend.git
cd hireme-ai-backend
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit your `.env` file to GitHub.

## Run Locally

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## API

### Health Check

```http
GET /
```

Response:

```json
{
  "message": "Resume Parsed!"
}
```

### Chat

```http
POST /chat
```

Request:

```json
{
  "question": "What are your strongest skills?"
}
```

The endpoint streams the AI response back to the frontend.

## Deployment

The backend can be deployed on platforms such as Render or Railway.

For a production deployment, use:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Set the `GROQ_API_KEY` environment variable in the hosting platform.

## Frontend

The HireMe AI frontend is built separately using React and Vite and can be deployed on Vercel.

```text
React Frontend
      ↓
FastAPI Backend
      ↓
Groq API
```

## Security

* API keys are stored using environment variables.
* `.env` should never be committed to the repository.
* CORS should be restricted to trusted frontend domains in production.

## Author

**Ayush Mahale**

B.Tech Computer Science Engineering

Built as an AI-powered resume and recruiter interview assistant.
