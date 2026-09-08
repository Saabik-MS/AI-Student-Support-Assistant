from flask import Flask, render_template, request, jsonify
from rag import answer_question

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()

    question = data.get("question", "").strip()

    if not question:
        return jsonify({
            "answer": "Please enter a question."
        })

    try:
        answer = answer_question(question)

        return jsonify({
            "answer": answer
        })

    except Exception as error:

        return jsonify({
            "answer": f"Error: {error}"
        })


if __name__ == "__main__":
    app.run(debug=True)