import uuid
from workflow.workflow import create_workflow
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize workflow
app = create_workflow()
config = {"configurable": {"thread_id": str(uuid.uuid1())}}

# Key test questions
test_questions = [
    "What is the green hydrogen cost for the user?",
    "How does CCS technology work on ships?",
    "What are the different generations of bioethanol feedstocks?",
]

print("\n" + "="*80)
print("QUICK RAG FUSION TEST")
print("="*80)

for idx, question in enumerate(test_questions, 1):
    print(f"\n{'='*80}")
    print(f"TEST {idx}/3: {question}")
    print(f"{'='*80}\n")

    try:
        result = app.invoke({"question": question}, config=config)

        if 'answer' in result:
            print(f"✅ ANSWER:\n{result['answer']}\n")
        elif 'error' in result:
            print(f"❌ ERROR: {result['error']}\n")

        if 'context' in result:
            print(f"📚 Retrieved {len(result['context'])} documents")

    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")

print("\n" + "="*80)
print("QUICK TEST COMPLETE")
print("="*80)
