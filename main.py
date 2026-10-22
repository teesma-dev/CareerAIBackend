
import json

from fastapi import FastAPI
from ollama import chat
from pydantic import BaseModel
from fastapi import HTTPException

app = FastAPI()


class ResumeRequest(BaseModel):
    resumeText: str


class ResumeSection(BaseModel):
    name: str
    score: int
    feedback: str


class ResumeAnalysis(BaseModel):
    overallScore: int
    summary: str
    strengths: list[str]
    missingKeywords: list[str]
    suggestions: list[str]
    sections: list[ResumeSection]

    
class JobDescriptionRequest(BaseModel):
    jobDescription: str
    resumeText: str


class JobAnalysis(BaseModel):
    roleTitle: str
    matchScore: int
    matchedSkills: list[str]
    skillsToImprove: list[str]

class InterviewRequest(BaseModel):
    question: str
    answer: str


class InterviewAnalysis(BaseModel):
    technicalAccuracy: int
    communication: int
    missingConcepts: list[str]
    suggestedAnswer: str
    overallFeedback: str

class InterviewQuestionRequest(BaseModel):
    topic: str = "SwiftUI"
    difficulty: str = "Intermediate"
    previousQuestions: list[str] = []


class InterviewQuestionResponse(BaseModel):
    question: str


class CodingQuestionRequest(BaseModel):
    difficulty: str = "Intermediate"
    language: str = "Swift"


class CodingQuestionResponse(BaseModel):
    title: str
    problem: str
    inputFormat: str
    outputFormat: str
    examples: list[str]


class CodingEvaluationRequest(BaseModel):
    problem: str
    language: str
    code: str


class CodingEvaluationResponse(BaseModel):
    correctness: int
    timeComplexity: str
    spaceComplexity: str
    codeQuality: int
    edgeCases: list[str]
    overallFeedback: str


@app.get("/")
def root():
    return {
        "message": "CareerAI backend is running with Ollama"
    }


@app.post("/analyze-resume", response_model=ResumeAnalysis)
def analyze_resume(request: ResumeRequest):

    response = chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": f"""
Analyze the following resume and return a complete analysis.

Resume:
{request.resumeText}

Analyze these resume sections individually:
1. Contact Information
2. Professional Summary
3. Work Experience
4. Skills
5. Education
6. Projects

For each section, return:
- name: the section name
- score: an integer from 0 to 100
- feedback: a short explanation of its quality or what needs improvement

If a section is missing, give it an appropriate score and explain what is missing.

Return valid JSON with exactly these top-level fields:
overallScore, summary, strengths, missingKeywords, suggestions, sections.

The sections field must be an array of objects, each containing
name, score, and feedback.

Also include:
- overallScore: integer from 0 to 100
- summary: a brief resume summary
- strengths: an array of strings
- missingKeywords: an array of strings
- suggestions: an array of strings

Ensure all scores are integers from 0 to 100.
Return JSON only, without Markdown or extra text.
"""
            }
        ],
        format="json"
    )

    result = json.loads(response.message.content)
    
    result.setdefault("overallScore", 0)
    result.setdefault("summary", "No summary available.")
    result.setdefault("strengths", [])
    result.setdefault("missingKeywords", [])
    result.setdefault("suggestions", [])
    result.setdefault("sections", [])


    return ResumeAnalysis(**result)


@app.post("/analyze-job-description", response_model=JobAnalysis)
def analyze_job_description(request: JobDescriptionRequest):

    response = chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": f"""
Compare the candidate's resume with the job description.

JOB DESCRIPTION:
{request.jobDescription}

CANDIDATE RESUME:
{request.resumeText}

Return valid JSON with exactly these fields:
- roleTitle: the job title
- matchScore: integer from 0 to 100
- matchedSkills: array of required skills supported by evidence
  in the candidate's resume
- skillsToImprove: array of job requirements not demonstrated
  in the candidate's resume

Rules:
- Only mark a skill as matched when the resume supports it.
- Do not assume the candidate has a skill just because the
  job description mentions it.
- Avoid duplicate skills.
- Use concise skill names.
- Return JSON only, without Markdown.
"""
            }
        ],
        format="json"
    )

    result = json.loads(response.message.content)

    result.setdefault("roleTitle", "Job Opportunity")
    result.setdefault("matchScore", 0)
    result.setdefault("matchedSkills", [])
    result.setdefault("skillsToImprove", [])

    result["matchScore"] = max(
        0, min(100, int(result["matchScore"]))
    )

    return JobAnalysis(**result)

@app.post(
    "/evaluate-interview-answer",
    response_model=InterviewAnalysis
)
def evaluate_interview_answer(
    request: InterviewRequest
):
    if not request.answer.strip():
        raise HTTPException(
            status_code=400,
            detail="Please provide an answer."
        )

    prompt = f"""
You are an experienced iOS technical interviewer.

Evaluate the candidate's answer to the following question.

QUESTION:
{request.question}

CANDIDATE ANSWER:
{request.answer}

Evaluate these areas:

1. technicalAccuracy: Integer from 0 to 100.
   Assess whether the answer is technically correct.

2. communication: Integer from 0 to 100.
   Assess clarity, organization, and explanation.

3. missingConcepts: List of important concepts the candidate
   missed or explained incompletely.

4. suggestedAnswer: A technically correct, interview-ready
   answer that explains the concept clearly.

5. overallFeedback: Constructive feedback explaining what
   the candidate did well and what should be improved.

For the question about @State and @Binding, consider:
- @State stores view-owned, mutable state.
- @Binding provides a read/write connection to state
  owned elsewhere.
- Changes to state can trigger SwiftUI view updates.
- A child view can use a binding to read and modify
  the parent's state.
- @State and @Binding are property wrappers used in
  different ownership situations.

Do not give full marks automatically.
Evaluate the candidate's actual answer fairly.
Do not assume they mentioned concepts that are absent.

Return ONLY valid JSON with exactly these keys:
{{
  "technicalAccuracy": 0,
  "communication": 0,
  "missingConcepts": [],
  "suggestedAnswer": "",
  "overallFeedback": ""
}}
"""

    try:
        response = chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format=InterviewAnalysis.model_json_schema(),
            options={
                "temperature": 0
            }
        )

        result = json.loads(response.message.content)

        # Validate the AI response before returning it.
        validated_result = InterviewAnalysis(**result)

        # Keep scores within the expected range.
        validated_result.technicalAccuracy = max(
            0, min(100, validated_result.technicalAccuracy)
        )
        validated_result.communication = max(
            0, min(100, validated_result.communication)
        )

        return validated_result

    except Exception as error:
        print("Interview evaluation error:", error)

        raise HTTPException(
            status_code=500,
            detail="Unable to evaluate the interview answer."
        )


@app.post(
    "/generate-interview-question",
    response_model=InterviewQuestionResponse
)
def generate_interview_question(
    request: InterviewQuestionRequest
):
    previous_questions = "\n".join(
        request.previousQuestions
    ) or "None"

    prompt = f"""
You are an experienced iOS technical interviewer.

Generate ONE interview question.

Topic: {request.topic}
Difficulty: {request.difficulty}

Previously asked questions:
{previous_questions}

Requirements:
- Ask a fresh, relevant technical question.
- Do not repeat a previously asked question.
- Match the requested topic and difficulty.
- Ask only one question.
- Do not provide the answer or explanation.
- Return valid JSON with exactly one key: question.
"""

    try:
        response = chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format=InterviewQuestionResponse.model_json_schema(),
            options={"temperature": 0.7}
        )

        result = json.loads(response.message.content)

        return InterviewQuestionResponse(**result)

    except Exception as error:
        print("Question generation error:", error)

        raise HTTPException(
            status_code=500,
            detail="Unable to generate an interview question."
        )

@app.post(
    "/generate-coding-question",
    response_model=CodingQuestionResponse
)
def generate_coding_question(
    request: CodingQuestionRequest
):

    prompt = f"""
You are an experienced technical interviewer.

Generate ONE coding interview problem.

Difficulty: {request.difficulty}
Programming Language: {request.language}

Requirements:
- The problem should be appropriate for the requested difficulty.
- It should be solvable using the requested programming language.
- Focus on common technical interview concepts.
- Do not provide the solution or code.
- Provide a clear problem statement.
- Include input format.
- Include output format.
- Include 2 or 3 examples.
- Return valid JSON with exactly these keys:
  title
  problem
  inputFormat
  outputFormat
  examples
"""

    try:
        response = chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format=CodingQuestionResponse.model_json_schema(),
            options={"temperature": 0.7}
        )

        result = json.loads(response.message.content)

        return CodingQuestionResponse(**result)

    except Exception as error:
        print("Coding question generation error:", error)

        raise HTTPException(
            status_code=500,
            detail="Unable to generate a coding question."
        )


@app.post(
    "/evaluate-coding-answer",
    response_model=CodingEvaluationResponse
)
def evaluate_coding_answer(
    request: CodingEvaluationRequest
):

    prompt = f"""
You are an experienced technical interviewer evaluating a coding interview answer.

Problem:
{request.problem}

Programming Language:
{request.language}

Candidate Code:
{request.code}

Evaluate the candidate's solution.

Requirements:

1. correctness
- Score from 0 to 100.
- Evaluate ONLY whether the submitted code correctly solves the given problem.
- If the solution correctly produces the expected result for the stated requirements and important edge cases, give 100.
- Do NOT reduce correctness because of variable naming, comments, coding style, implementation preferences, or possible optimizations.
- Do NOT reduce correctness simply because another implementation could be more efficient.
- Give less than 100 only when the code has an actual functional bug, produces incorrect results, or fails an important requirement of the problem.

2. timeComplexity
- Give the estimated Big-O time complexity.
- Example: O(n)

3. spaceComplexity
- Give the estimated Big-O space complexity.
- Example: O(n)

4. codeQuality
- Score from 0 to 100.
- Evaluate readability, naming, structure, maintainability, language conventions, and implementation quality.
- Minor optimization opportunities should affect codeQuality rather than correctness.

5. edgeCases
- List important edge cases the candidate should consider.

6. overallFeedback
- Give concise and useful interview feedback.
- Mention what was done well.
- Mention what could be improved.

Return valid JSON with exactly these keys:
correctness
timeComplexity
spaceComplexity
codeQuality
edgeCases
overallFeedback
"""

    try:
        response = chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format=CodingEvaluationResponse.model_json_schema(),
            options={"temperature": 0}
        )

        result = json.loads(response.message.content)

        return CodingEvaluationResponse(**result)

    except Exception as error:
        print("Coding evaluation error:", error)

        raise HTTPException(
            status_code=500,
            detail="Unable to evaluate coding answer."
        )

