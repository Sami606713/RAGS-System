from agent.agent import RunAgent

if __name__ == "__main__":
    query = """
Compare the N₂O emission factors between Jet Fuel Aircraft and Aviation Gasoline Aircraft from Table 5. Which aircraft type produces higher N₂O emissions per gallon of fuel?
"""
    response = RunAgent(query=query)
    print(">> Response: ",response)