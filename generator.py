from agent.agent import RunAgent

if __name__ == "__main__":
    query = """
Explain how the crystalline structure of cellulose impacts its enzymatic hydrolysis. What methods are used to overcome this challenge?
"""
    response = RunAgent(query=query)
    print(">> Response: ",response)