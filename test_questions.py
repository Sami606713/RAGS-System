"""
Test Questions for RAG System
Based on embedded documents in the vector database

Documents:
1. Charting-a-Greener-Course-Embrace-CCS-on-Maritime-Vessels.json
2. Cheat Sheet Hydrogen (1).json
3. Clean Energy Market Analysis in the US.json
4. Clean investment US.json
5. Comparative Life Cycle Assessment of Bioethanol Production from Different Generations ofBiomass and Waste Feedstocks.json
6. fuel_comparison.json
7. Green conversion of municipal solid wastes into fuels and chemicals.extraction.json
8. Waste to Fuel in NY.extraction.json
"""

import uuid
from workflow.workflow import create_workflow
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Test questions organized by topic
test_questions = {
    "Hydrogen Related": [
        "What is the green hydrogen cost for the user?",
        "What are the main production methods for hydrogen?",
        "How does green hydrogen compare to blue and gray hydrogen?",
        "What are the applications of hydrogen in clean energy?",
        "What is the efficiency of hydrogen fuel cells?",
        "What are the storage challenges for hydrogen?",
    ],

    "Carbon Capture (CCS) on Maritime Vessels": [
        "What is carbon capture and storage on maritime vessels?",
        "How does CCS technology work on ships?",
        "What are the benefits of implementing CCS on maritime vessels?",
        "What are the challenges of CCS implementation in the maritime industry?",
        "Which ships are best suited for CCS technology?",
    ],

    "Clean Energy Market in US": [
        "What is the current state of clean energy market in the United States?",
        "What are the key trends in US clean energy investment?",
        "Which states are leading in clean energy adoption?",
        "What renewable energy sources dominate the US market?",
        "What are the growth projections for clean energy in the US?",
    ],

    "Bioethanol Production": [
        "What are the different generations of bioethanol feedstocks?",
        "How do first, second, and third generation bioethanol compare?",
        "What is the life cycle assessment of bioethanol production?",
        "What are the environmental impacts of bioethanol from different biomass sources?",
        "Which bioethanol production method is most sustainable?",
    ],

    "Fuel Comparison": [
        "How do different clean fuels compare in terms of cost?",
        "What are the efficiency differences between various alternative fuels?",
        "Which fuel has the lowest carbon footprint?",
        "How does the energy density of different fuels compare?",
        "What are the pros and cons of biodiesel vs bioethanol?",
    ],

    "Waste to Fuel": [
        "How can municipal solid waste be converted to fuel?",
        "What technologies are used for waste to fuel conversion?",
        "What are the benefits of converting waste to energy in New York?",
        "What types of waste can be converted to fuel?",
        "What is the efficiency of waste to fuel conversion processes?",
    ],

    "Clean Investment": [
        "What is the current level of clean energy investment in the US?",
        "Which sectors receive the most clean energy funding?",
        "What are the investment trends in renewable energy?",
        "How has clean energy investment changed over the past decade?",
        "What role does government policy play in clean energy investment?",
    ],

    "Comparative/Complex Questions": [
        "Compare the costs of green hydrogen vs traditional fossil fuels",
        "What is the most cost-effective clean energy solution for transportation?",
        "How do carbon capture costs compare to renewable energy investments?",
        "Which waste-to-fuel technology has the best ROI?",
        "What are the economic and environmental trade-offs of different bioethanol generations?",
    ]
}

def run_test_questions(questions_dict, max_per_category=3):
    """Run test questions through the RAG system"""

    # Initialize workflow
    app = create_workflow()
    config = {"configurable": {"thread_id": str(uuid.uuid1())}}

    print("\n" + "="*80)
    print("RAG SYSTEM TEST QUESTIONS")
    print("="*80)

    for category, questions in questions_dict.items():
        print(f"\n{'='*80}")
        print(f"CATEGORY: {category}")
        print(f"{'='*80}\n")

        # Test first N questions from each category
        for idx, question in enumerate(questions[:max_per_category], 1):
            print(f"\n{'-'*80}")
            print(f"Q{idx}: {question}")
            print(f"{'-'*80}")

            try:
                result = app.invoke({"question": question}, config=config)

                if 'answer' in result:
                    # Truncate long answers for readability
                    answer = result['answer']
                    if len(answer) > 300:
                        answer = answer[:300] + "..."
                    print(f"Answer: {answer}")
                elif 'error' in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"Result: {result}")

            except Exception as e:
                print(f"Exception: {str(e)}")

            print()  # Blank line between questions

if __name__ == "__main__":
    # Run all test questions (3 per category by default)
    run_test_questions(test_questions, max_per_category=3)

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
