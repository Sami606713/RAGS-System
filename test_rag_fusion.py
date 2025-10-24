import uuid
from workflow.workflow import create_workflow
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize workflow app
app = create_workflow()
config = {
    "configurable": {"thread_id": str(uuid.uuid1())},
}

# Test query
user_input = "What is the green hydrogen cost?"

print(f"\n{'='*60}")
print(f"Question: {user_input}")
print(f"{'='*60}\n")

# Run the workflow
try:
    result = app.invoke({"question": user_input}, config=config)

    print("\n" + "="*60)
    print("RESULT:")
    print("="*60)

    if 'answer' in result:
        print(f"\nAnswer: {result['answer']}\n")
    elif 'error' in result:
        print(f"\nError: {result['error']}\n")
    else:
        print(f"\nRaw result: {result}\n")

    # Print context info
    if 'context' in result and result['context']:
        print(f"\nRetrieved {len(result['context'])} context documents")

except Exception as e:
    print(f"\nException occurred: {str(e)}")
    import traceback
    traceback.print_exc()
