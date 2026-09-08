import os
import re
import json
import ollama

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


# ============================================================
# MEMORY
# ============================================================

memory = {}


def load_memory():
    """Load saved student memory from memory.json."""
    try:
        with open("memory.json", "r", encoding="utf-8") as file:
            return json.load(file)

    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_memory(data):
    """Save student memory to memory.json."""
    with open("memory.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def save_chat(question, answer):
    """Save the latest chat conversations."""
    memory.setdefault("chat_history", [])

    memory["chat_history"].append({
        "question": question,
        "answer": answer
    })

    # Keep only the last 5 conversations
    memory["chat_history"] = memory["chat_history"][-5:]

    save_memory(memory)


# ============================================================
# CGPA TOOL
# ============================================================

def calculate_cgpa(marks):
    """Calculate simple average of entered CGPA/marks values."""
    if not marks:
        return None

    return round(sum(marks) / len(marks), 2)


# ============================================================
# ATTENDANCE TOOL
# ============================================================

def calculate_attendance(attended, total):
    """Calculate attendance percentage."""
    if total <= 0:
        return None

    return round((attended / total) * 100, 2)


# ============================================================
# LOAD DOCUMENTS
# ============================================================

documents = []

document_folder = "documents"

if not os.path.exists(document_folder):
    raise FileNotFoundError(
        "ERROR: 'documents' folder not found."
    )


for filename in os.listdir(document_folder):

    file_path = os.path.join(
        document_folder,
        filename
    )

    if filename.lower().endswith(".txt"):

        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )

        documents.extend(loader.load())


print("Documents loaded:", len(documents))


if not documents:
    raise ValueError(
        "ERROR: No .txt files found inside 'documents' folder."
    )


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(
    documents
)

print("Chunks created:", len(chunks))


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("RAG database ready!")


# ============================================================
# LOAD MEMORY
# ============================================================

memory = load_memory()


# ============================================================
# COLLEGE / INSTITUTION KEYWORDS
# ============================================================

college_keywords = [
    "college",
    "department",
    "student",
    "syllabus",
    "exam",
    "examination",
    "attendance",
    "assignment",
    "laboratory",
    "lab",
    "project",
    "faculty",
    "regulation",
    "course",
    "semester",
    "academic",
    "internship",
    "schedule",
    "notice"
]


# ============================================================
# SIMPLE CONVERSATION
# ============================================================

simple_messages = {
    "hi": "Hello! How can I help you with your studies?",
    "hello": "Hello! How can I help you with your studies?",
    "hey": "Hi! What would you like to learn today?",
    "ok": "Sure! Ask me any study-related question.",
    "okay": "Sure! Ask me any study-related question.",
    "thanks": "You're welcome! Ask me another question.",
    "thank you": "You're welcome! Ask me another question."
}


# ============================================================
# MAIN ANSWER FUNCTION
# ============================================================

def answer_question(question):

    question = question.strip()

    if not question:
        return "Please enter a question."


    question_lower = question.lower()


    # ========================================================
    # SIMPLE MESSAGES
    # ========================================================

    if question_lower in simple_messages:

        answer = simple_messages[question_lower]

        save_chat(
            question,
            answer
        )

        return answer


    # ========================================================
    # EXIT
    # ========================================================

    if question_lower == "exit":

        return "Goodbye!"


    # ========================================================
    # SAVE NAME
    # ========================================================

    if "my name is" in question_lower:

        position = question_lower.index(
            "my name is"
        )

        name = question[
            position + len("my name is"):
        ].strip()

        if not name:

            return "Please tell me your name."


        memory["name"] = name

        save_memory(memory)

        answer = (
            f"Nice to meet you {name}. "
            f"I will remember your name."
        )

        return answer


    # ========================================================
    # GET NAME
    # ========================================================

    if (
        "what is my name" in question_lower
        or "what's my name" in question_lower
        or "do you remember my name" in question_lower
    ):

        if "name" in memory:

            return (
                f"Your name is "
                f"{memory['name']}."
            )

        return (
            "You have not told me "
            "your name yet."
        )


    # ========================================================
    # SAVE DEPARTMENT
    # ========================================================

    if "my department is" in question_lower:

        position = question_lower.index(
            "my department is"
        )

        department = question[
            position + len("my department is"):
        ].strip()

        if not department:

            return "Please tell me your department."


        memory["department"] = department

        save_memory(memory)

        return (
            f"Okay, I will remember that "
            f"your department is {department}."
        )


    # ========================================================
    # GET DEPARTMENT
    # ========================================================

    if (
        "what department" in question_lower
        or "which department" in question_lower
        or "what is my department" in question_lower
        or "do you remember my department" in question_lower
    ):

        if "department" in memory:

            return (
                f"You are studying "
                f"{memory['department']}."
            )

        return (
            "You have not told me "
            "your department yet."
        )


    # ========================================================
    # ATTENDANCE TOOL
    # ========================================================

    attendance_keywords = [
        "attendance",
        "attended",
        "classes attended"
    ]

    is_attendance_question = any(
        word in question_lower
        for word in attendance_keywords
    )

    attendance_numbers = re.findall(
        r"\d+(?:\.\d+)?",
        question
    )

    if (
        is_attendance_question
        and len(attendance_numbers) >= 2
    ):

        attended = float(
            attendance_numbers[0]
        )

        total = float(
            attendance_numbers[1]
        )

        if total <= 0:

            return (
                "Total classes must be greater than zero."
            )

        if attended > total:

            return (
                "Attended classes cannot be "
                "greater than total classes."
            )

        result = calculate_attendance(
            attended,
            total
        )

        answer = (
            f"Your attendance is {result}%."
        )

        save_chat(
            question,
            answer
        )

        return answer


    # ========================================================
    # CGPA TOOL
    # ========================================================

    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        question
    )

    is_cgpa_question = (
        "cgpa" in question_lower
        or "marks" in question_lower
        or "grade" in question_lower
    )

    if (
        len(numbers) >= 2
        and is_cgpa_question
    ):

        marks = [
            float(number)
            for number in numbers
        ]

        result = calculate_cgpa(
            marks
        )

        answer = (
            f"Your calculated CGPA is {result}."
        )

        save_chat(
            question,
            answer
        )

        return answer


    # ========================================================
    # DECIDE WHETHER TO USE RAG
    # ========================================================

    is_college_question = any(
        keyword in question_lower
        for keyword in college_keywords
    )


    # ========================================================
    # RAG SEARCH
    # ========================================================

    context = ""

    if is_college_question:

        try:

            results = vectorstore.similarity_search(
                question,
                k=3
            )

            context = "\n\n".join(
                result.page_content
                for result in results
            )

        except Exception as error:

            print("RAG search error:", error)

            context = ""


    # ========================================================
    # GENERAL STUDY PROMPT
    # ========================================================

    if not is_college_question:

        prompt = f"""
You are an AI Student Support Assistant and study tutor.

Answer the student's question directly.

The student wants to understand the topic clearly.

For academic or technical questions, use this structure
when appropriate:

Definition:
Simple Explanation:
Example:
Key Points:

Use simple English.

Do not act as the student.
Do not write "Student:".
Do not ask another question.
Do not change the topic.

Student Question:
{question}

Give only the useful answer.
"""


    # ========================================================
    # COLLEGE RAG PROMPT
    # ========================================================

    else:

        prompt = f"""
You are an AI Student Support Assistant.

Answer the student's question using the relevant
college information provided below.

IMPORTANT RULES:

1. Use only information relevant to the question.
2. Ignore unrelated information.
3. Do not copy the document word-for-word.
4. Do not invent college-specific facts.
5. Do not act as the student.
6. Do not write "Student:".
7. Do not ask another question.
8. Answer directly and clearly.

College Information:
{context}

Student Question:
{question}

Give only the answer to the student's question.
"""


    # ========================================================
    # GENERATE ANSWER USING OLLAMA
    # ========================================================

    try:

        response = ollama.chat(
            model="tinyllama:latest",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI student tutor. "
                        "Answer questions directly, clearly, "
                        "and simply."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"].strip()


        # ====================================================
        # CLEAN UNWANTED PREFIXES
        # ====================================================

        unwanted_prefixes = [
            "student:",
            "assistant:",
            "ai:"
        ]

        for prefix in unwanted_prefixes:

            if answer.lower().startswith(prefix):

                answer = answer[
                    len(prefix):
                ].strip()


        # ====================================================
        # SAVE CHAT
        # ====================================================

        save_chat(
            question,
            answer
        )


        return answer


    except Exception as error:

        return (
            "Sorry, I could not generate an answer.\n"
            f"Error: {error}"
        )