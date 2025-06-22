from agent.agent import RunAgent

# Simple Questions
simple_questions = [
    "What does WPT stand for in the context of maritime shipping?",
    "What is the primary goal of adopting wind-assisted ship propulsion technologies?",
    "Name three types of wind propulsion technologies discussed in the paper.",
    "What is the main function of the International Maritime Organization (IMO)?",
    "What is the maximum vessel lifetime considered in the agent-based model?"
]

# Medium-Level Questions
medium_questions = [
    "Explain how the Bass model is used to simulate awareness of wind propulsion technology.",
    "Describe the role of the utility-based decision-making mechanism in WPT adoption.",
    "What is the significance of the social network structure in the agent-based simulation?",
    "How does the installation cost (𝐾𝑗) affect the adoption decision in the model?",
    "Compare the advantages of Flettner rotors and Wingsail technologies in terms of installation and fuel savings."
]

# Complex Questions
complex_questions = [
    "Why does increased WPT awareness lead to more Ventifoil adoption compared to Wingsail in mid-subsidy scenarios?",
    "How do network effects influence early adoption in the simulation, and why is SF4 chosen as the topology?",
    "Under what policy conditions does Wingsail adoption outperform Ventifoil in the 30-year forecast?",
    "Explain why subsidies have a more significant impact than fuel tax in promoting WPT adoption.",
    "How does heterogeneity in vessel characteristics affect the utility calculation across different WPT options?"
]

# Graph/Chart-Based Questions (without figure references)
graph_questions = [
    "How does initial awareness percentage affect the shape of the awareness diffusion curve?",
    "Which WPT technology shows the highest adoption rate at high sailing distances and why?",
    "At what installation subsidy level does the Ventifoil option start to outperform Wingsail adoption?",
    "What trend is observed when initial awareness is increased to 10%? How does this affect technology preference?",
    "How does the combined impact of subsidy and fuel tax change the long-term adoption trend of Ventifoil?"
]


def generate_responses(category_name, questions):
    responses = []
    for i, query in enumerate(questions, 1):
        print(f">> Generating response for {category_name} Q{i}: {query}")
        response = RunAgent(query=query)
        responses.append((query, response))
    return responses

def save_to_txt(file_path, all_responses):
    with open(file_path, 'w', encoding='utf-8') as f:
        for category, responses in all_responses.items():
            f.write(f"\n=== {category.upper()} QUESTIONS ===\n\n")
            for i, (q, a) in enumerate(responses, 1):
                f.write(f"Q{i}: {q}\n")
                f.write(f"A{i}: {a}\n\n")

if __name__ == "__main__":
    all_responses = {
        "Simple": generate_responses("Simple", simple_questions),
        "Medium": generate_responses("Medium", medium_questions),
        "Complex": generate_responses("Complex", complex_questions),
        "Graph-Based": generate_responses("Graph-Based", graph_questions)
    }

    save_to_txt("wind-assist.txt", all_responses)
    print("✅ All responses saved to responses.txt")
