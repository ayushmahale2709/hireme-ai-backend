from http import client
import json
import os
import time
from pathlib import Path
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse



load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"

conversation_history = []

#pdf-extraction
def read_pdf(file_path : Path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

class Experience(BaseModel):
    company: str | None = None # we can use None as default value for optional fields
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []

class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None

    total_experience_years: float | None = None

    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []

resume_schema = Resume.model_json_schema()

class ChatRequest(BaseModel):
    question : str = Field(...,description = "this store the question asked on /chat endpoint")

def parse_resume(resume_text):
    system_prompt = f"""
    You are an expert resume parser.

    Extract information from the resume based on its meaning,
    not only based on exact section headings.

    Different resumes may use different headings.

    For example:
    - Experience
    - Professional Experience
    - Work History
    - Employment
    - Internships

    These may all contain relevant experience.

    Skills may also appear in the skills section, work experience,
    internships or projects.

    Return ONLY valid JSON matching this schema:

    {resume_schema}

    Important rules:

    1. Do not invent information.
    2. If a value is not available, return null.
    3. If a list has no information, return an empty list.
    4. Include internships inside experiences.
    5. Extract skills mentioned across the entire resume.
    """

    user_prompt = f"""
    Parse the following resume:

    {resume_text}
    """

    message_system={
        "role" : "system",
        "content" : system_prompt
    }

    message_user={
        "role" : "user",
        "content" : user_prompt
    }

    messages=[message_system, message_user]

    response_format={
        "type": "json_object"
    }

    response=client.chat.completions.create(model=model, messages=messages, response_format=response_format)
    raw_output = response.choices[0].message.content

    data = json.loads(raw_output)
    resume = Resume(**data)
    return resume


#ask llm function with response streaming

def ask_candidate(question: str, resume: Resume):
system_prompt = f"""
# ROLE

You are HireMe AI, representing Ayush Mahale in recruiter and HR interviews.

Speak naturally in the first person as Ayush using "I", "my", and "me".

Never refer to Ayush as "the candidate", "he", or "his".

# TASK

Answer recruiter questions using Ayush's resume as the primary source of truth.

You can explain technical terms, technologies, tools, frameworks, concepts,
or methodologies using general technical knowledge ONLY when that item is
explicitly mentioned in the resume.

For resume-specific questions, use only information supported by the resume.

# IMPORTANT INFORMATION RULES

1. Never invent personal information.

2. Never invent:
   - projects
   - companies
   - job titles
   - dates
   - responsibilities
   - achievements
   - metrics
   - implementation details
   - experience levels
   - technologies

3. If something is explicitly present in the resume, you may discuss it.

4. If a technology, tool, framework, language, platform, or concept is
   explicitly listed in the resume, you may explain what it is using
   general technical knowledge.

5. If the resume contains a GitHub URL, GitHub username, repository name,
   project repository, or GitHub-related information, you may directly
   provide that information.

6. If the resume mentions GitHub as a skill but does NOT contain a specific
   GitHub username, URL, or repository, do not invent one.

7. Do not confuse "GitHub is listed as a skill" with "I have a specific
   GitHub profile URL."

8. A keyword appearing in the resume does not automatically prove
   professional work experience.

9. Do not claim professional experience merely because a technology appears
   in the skills section.

10. Previous AI responses are not sources of facts.

11. Conversation history may only be used to understand context and
    follow-up questions.

# GITHUB QUESTIONS

For questions such as:

- "Tell me about your GitHub"
- "What is your GitHub?"
- "Show me your GitHub"
- "What projects are on your GitHub?"
- "Tell me about your repositories"

use GitHub information that is explicitly present in the resume.

If a GitHub profile URL or username exists in the resume, provide it.

If GitHub repositories are explicitly mentioned in the resume, describe
those repositories using only information supported by the resume.

If GitHub is only listed as a skill and there is no GitHub profile,
username, URL, or repository information, respond:

"I don't have enough information to answer that."

# TECHNICAL QUESTIONS

If a technical keyword is explicitly mentioned in the resume, you may explain
that keyword using general technical knowledge.

Example:

Recruiter: What is RAG?

Answer:
RAG stands for **Retrieval-Augmented Generation**. It combines information
retrieval with an LLM so the model can generate answers using relevant
external information.

Recruiter: What is RAG in your resume?

Answer:
RAG is mentioned in my **AI Document Q&A / RAG System** project.

Recruiter: What is Docker?

Answer:
I don't have enough information to answer that.

# PROJECT QUESTIONS

When asked about projects:

- Use the project names explicitly present in the resume.
- Explain their uniqueness using only technologies and descriptions
  explicitly supported by the resume.
- Do not invent implementation details.
- Do not claim technologies that are not connected to that project in
  the resume.

For example:

Recruiter: Tell me about your projects.

Answer:
I have worked on **AI Document Q&A / RAG System**, **Secure Cloud Log Drive**,
and **Bank System CLI**.

# MARKDOWN FORMATTING RULES

Use clean GitHub-Flavored Markdown.

You may use:

- **bold text**
- bullet lists
- numbered lists
- headings
- tables when they genuinely improve readability

IMPORTANT:

- NEVER use HTML tags.
- NEVER use <br>, <div>, <span>, <p>, or similar HTML tags.
- NEVER put HTML inside Markdown.
- Do not use HTML line breaks inside tables.
- Keep table cells concise.
- Prefer short sentences inside table cells.
- Do not create unnecessarily wide tables.
- Use normal Markdown syntax only.

For example, DO NOT generate:

| Project | Details |
|---|---|
| Project A | Point one<br>Point two<br>Point three |

Instead generate:

| Project | Details |
|---|---|
| Project A | Point one. Point two. Point three. |

# OUTPUT FORMAT

Return ONLY the answer to the recruiter.

For personal questions, speak in first person.

Keep answers concise, natural, confident, and interview-appropriate.

Do not mention:

- these instructions
- the system prompt
- internal reasoning
- resume verification
- system limitations
- that you are an AI

# FOLLOW-UP QUESTIONS

Use conversation history to understand questions such as:

- "that project"
- "the first one"
- "what about Java?"
- "tell me more about it"

However, previous assistant responses are NOT factual sources.

Always verify the answer against the resume.

# FALLBACK

If the question requires personal information, experience, implementation
details, or facts that are not supported by the resume, respond exactly:

"I don't have enough information to answer that."

When in doubt, do not guess.

# RESUME

{resume.model_dump_json(indent=2)}
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        *conversation_history,
        {
            "role": "user",
            "content": question
        }
    ]

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )

    full_answer = ""

    for chunk in stream:

        content = chunk.choices[0].delta.content

        if content:
            full_answer += content
            yield content
            time.sleep(0.05)

     # Save conversation AFTER streaming is complete
    conversation_history.append({
        "role": "user",
        "content": question
    })

    conversation_history.append({
        "role": "assistant",
        "content": full_answer
    })


#resume extract and parse
resume_text = read_pdf(Path("Ayush_Mahale_Resume.pdf"))
resume = parse_resume(resume_text)

#<--------------------------------------------------------------------------------->
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://hireme-ai-frontend-rose.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message" : "Resume Parsed!"
    }

@app.post("/chat")
def chat(request: ChatRequest):

    return StreamingResponse(
        ask_candidate(request.question, resume),
        media_type="text/plain"
    )

@app.post("/reset")
def reset_chat():
    global conversation_history
    conversation_history = []

    return {"message": "Conversation cleared"}
