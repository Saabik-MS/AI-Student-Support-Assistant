import ollama

print("=" * 50)
print("     AI STUDENT SUPPORT ASSISTANT")
print("=" * 50)
print("Type 'exit' to stop.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("AI: Goodbye!")
        break

    response = ollama.chat(
        model="tinyllama:latest",
        messages=[
            {
                "role": "system",
                "content": "You are an AI Student Support Assistant. Answer student questions clearly and simply."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    answer = response["message"]["content"]

    print("AI:", answer)
    print()
    