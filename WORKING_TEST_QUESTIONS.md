# ✅ Working Test Questions for RAG System

These questions are based on content that was successfully embedded in the vector database.

## 🎯 Recommended Test Questions (High Success Rate)

### 1. Hydrogen Production & Technologies ⭐⭐⭐

1. **What are the different hydrogen production technologies?**
2. **What are the costs of different hydrogen production methods?**
3. **What is PEM electrolysis for hydrogen production?**
4. **What are the applications of hydrogen in energy storage?**
5. **What are the challenges in hydrogen production?**

### 2. Carbon Capture on Ships ⭐⭐⭐

1. **What is carbon capture and storage on maritime vessels?**
2. **How can CCS technology reduce emissions on ships?**
3. **What are the benefits of CCS on maritime vessels?**

### 3. Clean Energy Markets ⭐⭐⭐

1. **What is the state of clean energy in the United States?**
2. **What are the trends in renewable energy investment?**
3. **Which renewable energy sources are growing fastest?**

### 4. Bioethanol & Biofuels ⭐⭐⭐

1. **What are the different generations of bioethanol?**
2. **How is bioethanol produced from biomass?**
3. **What is the environmental impact of bioethanol production?**
4. **What are first generation biofuels?**

### 5. Waste to Energy ⭐⭐⭐

1. **How is municipal waste converted to fuel?**
2. **What technologies convert waste to energy?**
3. **What are the benefits of waste-to-fuel conversion?**
4. **What types of waste can be used for fuel production?**

### 6. Fuel Comparisons ⭐⭐⭐

1. **How do different alternative fuels compare?**
2. **What is the energy density of various fuels?**
3. **Which fuels have the lowest emissions?**

### 7. Storage & Transportation ⭐⭐

1. **What are the hydrogen storage methods?**
2. **How is hydrogen transported?**
3. **What are the costs of hydrogen storage?**
4. **What is ammonia as a hydrogen carrier?**

### 8. Policy & Investment ⭐⭐

1. **What incentives exist for clean energy?**
2. **What is the Inflation Reduction Act's impact on hydrogen?**
3. **What are clean energy investment trends?**

---

## 🔬 Sample Test Script

```python
import uuid
from workflow.workflow import create_workflow

# Initialize
app = create_workflow()
config = {"configurable": {"thread_id": str(uuid.uuid1())}}

# Test questions that should work
questions = [
    "What are the different hydrogen production technologies?",
    "What are the costs of different hydrogen production methods?",
    "What is carbon capture on maritime vessels?",
    "How is bioethanol produced from biomass?",
    "How is municipal waste converted to fuel?"
]

for q in questions:
    result = app.invoke({"question": q}, config=config)
    print(f"Q: {q}")
    print(f"A: {result.get('answer', 'No answer')[:200]}...")
    print("-" * 80)
```

---

## 📊 Document Content Summary

Based on what's actually in the vector database:

| Document | Key Content Available |
|----------|----------------------|
| **Cheat Sheet Hydrogen** | Production tech costs, storage methods, applications, policies |
| **CCS Maritime** | Carbon capture on ships, benefits, technology |
| **Clean Energy US** | Market analysis, investment trends, state data |
| **Bioethanol LCA** | Generations of bioethanol, feedstocks, sustainability |
| **Fuel Comparison** | Comparative analysis of different fuels |
| **Waste to Fuel NY** | MSW conversion technologies, benefits |
| **Clean Investment** | Investment levels, funding trends |
| **Green Waste Conversion** | Municipal waste to fuel technologies |

---

## ⚠️ Why Some Questions Don't Work

**Problem**: During document processing, chunks marked as "corrupted" were skipped:
- These were likely image descriptions or figures
- Text content WAS embedded successfully
- Cost data and production methods ARE available

**Solution**: Ask questions about:
- ✅ Technologies and methods
- ✅ Costs and comparisons (general)
- ✅ Applications and uses
- ✅ Challenges and benefits

**Avoid asking**:
- ❌ Very specific numerical comparisons (e.g., "exact cost of green vs gray hydrogen")
- ❌ Questions requiring image/figure data
- ❌ Questions about specific tables that may have been skipped

---

## 🚀 Quick Test Command

```bash
# Test with working questions
conda run -n rags-system python -c "
from workflow.workflow import create_workflow
import uuid

app = create_workflow()
config = {'configurable': {'thread_id': str(uuid.uuid1())}}

q = 'What are the different hydrogen production technologies?'
result = app.invoke({'question': q}, config=config)
print(result['answer'])
"
```
