# Test Questions for RAG System

Based on the 8 embedded documents in the vector database.

## 📄 Embedded Documents:
1. Charting-a-Greener-Course-Embrace-CCS-on-Maritime-Vessels.json
2. Cheat Sheet Hydrogen (1).json
3. Clean Energy Market Analysis in the US.json
4. Clean investment US.json
5. Comparative Life Cycle Assessment of Bioethanol Production from Different Generations ofBiomass and Waste Feedstocks.json
6. fuel_comparison.json
7. Green conversion of municipal solid wastes into fuels and chemicals.extraction.json
8. Waste to Fuel in NY.extraction.json

---

## 🧪 Test Questions by Category

### 1. Hydrogen Related Questions

1. **What is the green hydrogen cost for the user?**
2. What are the main production methods for hydrogen?
3. How does green hydrogen compare to blue and gray hydrogen?
4. What are the applications of hydrogen in clean energy?
5. What is the efficiency of hydrogen fuel cells?
6. What are the storage challenges for hydrogen?

### 2. Carbon Capture (CCS) on Maritime Vessels

1. What is carbon capture and storage on maritime vessels?
2. How does CCS technology work on ships?
3. What are the benefits of implementing CCS on maritime vessels?
4. What are the challenges of CCS implementation in the maritime industry?
5. Which ships are best suited for CCS technology?

### 3. Clean Energy Market in US

1. What is the current state of clean energy market in the United States?
2. What are the key trends in US clean energy investment?
3. Which states are leading in clean energy adoption?
4. What renewable energy sources dominate the US market?
5. What are the growth projections for clean energy in the US?

### 4. Bioethanol Production

1. What are the different generations of bioethanol feedstocks?
2. How do first, second, and third generation bioethanol compare?
3. What is the life cycle assessment of bioethanol production?
4. What are the environmental impacts of bioethanol from different biomass sources?
5. Which bioethanol production method is most sustainable?

### 5. Fuel Comparison

1. How do different clean fuels compare in terms of cost?
2. What are the efficiency differences between various alternative fuels?
3. Which fuel has the lowest carbon footprint?
4. How does the energy density of different fuels compare?
5. What are the pros and cons of biodiesel vs bioethanol?

### 6. Waste to Fuel Conversion

1. How can municipal solid waste be converted to fuel?
2. What technologies are used for waste to fuel conversion?
3. What are the benefits of converting waste to energy in New York?
4. What types of waste can be converted to fuel?
5. What is the efficiency of waste to fuel conversion processes?

### 7. Clean Energy Investment

1. What is the current level of clean energy investment in the US?
2. Which sectors receive the most clean energy funding?
3. What are the investment trends in renewable energy?
4. How has clean energy investment changed over the past decade?
5. What role does government policy play in clean energy investment?

### 8. Comparative/Complex Questions

1. **Compare the costs of green hydrogen vs traditional fossil fuels**
2. What is the most cost-effective clean energy solution for transportation?
3. How do carbon capture costs compare to renewable energy investments?
4. Which waste-to-fuel technology has the best ROI?
5. What are the economic and environmental trade-offs of different bioethanol generations?

---

## 🚀 How to Test

### Option 1: Run Automated Test Script
```bash
conda run -n rags-system python test_questions.py
```

### Option 2: Test via Streamlit App
Visit http://localhost:8502 and paste questions one by one

### Option 3: Test Individual Questions
```python
from workflow.workflow import create_workflow
import uuid

app = create_workflow()
config = {"configurable": {"thread_id": str(uuid.uuid1())}}

result = app.invoke({"question": "YOUR_QUESTION_HERE"}, config=config)
print(result['answer'])
```

---

## 📊 Expected Results

- **RAG Fusion** should generate 5 query variations for each question
- System should retrieve and re-rank relevant documents
- Answers should include citations from source documents
- Complex comparative questions should synthesize info from multiple docs

---

## 🔍 Key Questions to Focus On:

1. ⭐ **What is the green hydrogen cost for the user?** (Main test question)
2. ⭐ **Compare the costs of green hydrogen vs traditional fossil fuels** (Complex comparison)
3. ⭐ **What is the most cost-effective clean energy solution for transportation?** (Multi-doc synthesis)
