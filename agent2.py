from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
CSV_PATH = os.path.join(os.path.dirname(__file__), "drugs_side_effects_drugs_com.csv")

def _load_csv() -> pd.DataFrame:
    """Load and clean the drugs CSV. Called once at import time."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found at: {CSV_PATH}\n"
            "Please place 'drugs_side_effects_drugs_com.csv' in the same folder as agent.py"
        )
    df = pd.read_csv(CSV_PATH)
    # Normalise text columns for case-insensitive search
    df["_drug_name_lower"]    = df["drug_name"].fillna("").str.lower().str.strip()
    df["_generic_name_lower"] = df["generic_name"].fillna("").str.lower().str.strip()
    df["_brand_names_lower"]  = df["brand_names"].fillna("").str.lower()
    df["_condition_lower"]    = df["medical_condition"].fillna("").str.lower().str.strip()
    return df
 
try:
    DF = _load_csv()
    print(f"✅ CSV loaded: {len(DF)} drug records")
except FileNotFoundError as e:
    print(f"⚠️  {e}")
    DF = pd.DataFrame()   # empty — tools will return helpful errors
 
# Global chat history
chat_history = []
 
# HELPER — find rows matching a medicine name
def _find_rows(medicine_name: str) -> pd.DataFrame:
    """
    Return all CSV rows where drug_name, generic_name, or brand_names
    match the query (case-insensitive, partial match allowed).
    Returns the best single match (exact > partial) as a 1-row DataFrame,
    or empty DataFrame if nothing found.
    """
    if DF.empty:
        return pd.DataFrame()
 
    q = medicine_name.lower().strip()
    def _best(subset: pd.DataFrame) -> pd.DataFrame:
        """Return the highest-rated row from a subset."""
        if "rating" in subset.columns:
            return subset.sort_values("rating", ascending=False).iloc[[0]]
        return subset.iloc[[0]]
 
    # 1. Exact drug_name match
    exact = DF[DF["_drug_name_lower"] == q]
    if not exact.empty:
        return _best(exact)
 
    # 2. Exact generic_name match
    exact_gen = DF[DF["_generic_name_lower"] == q]
    if not exact_gen.empty:
        return _best(exact_gen)
 
    # 3. Partial drug_name match
    partial = DF[DF["_drug_name_lower"].str.contains(q, na=False)]
    if not partial.empty:
        return _best(partial)
 
    # 4. Partial generic_name match
    partial_gen = DF[DF["_generic_name_lower"].str.contains(q, na=False)]
    if not partial_gen.empty:
        return _best(partial_gen)
 
    # 5. Brand name contains query
    brand = DF[DF["_brand_names_lower"].str.contains(q, na=False)]
    if not brand.empty:
        return _best(brand)
    return pd.DataFrame()
 
def _find_condition(symptom: str) -> pd.DataFrame:
    if DF.empty:
        return pd.DataFrame()
    q = symptom.lower().strip()
    return DF[DF["_condition_lower"].str.contains(q, na=False, regex=False)]
 
def _pregnancy_label(code: str) -> str:
    labels = {
        "A": "A — Safe (adequate studies show no risk)",
        "B": "B — Probably safe (animal studies OK, limited human data)",
        "C": "C — Use with caution (risk cannot be ruled out)",
        "D": "D — Positive evidence of risk — use only if benefits outweigh risks",
        "X": "X — CONTRAINDICATED in pregnancy",
        "N": "N — Not classified",
    }
    return labels.get(str(code).strip().upper(), code)

@tool
def get_medicine_info(medicine_name: str) -> str:
    """Returns uses, indications, dosage, and drug class for a given medicine name. Use this when the user
    asks what a specific medicine is used for."""
    if DF.empty:
        return "⚠️ Medicine information is currently unavailable. Please ensure the CSV file is correctly placed."
    try:
        if not medicine_name or not medicine_name.strip():
            return "⚠️ Please provide a valid medicine name."
        rows = _find_rows(medicine_name)
        if rows.empty:
            return f"⚠️ No information found for medicine: '{medicine_name}'. Please check the name and try again."
        row = rows.iloc[0]
        return (
            f"💊 {row['drug_name']}\n"
            f"   Used For   : {row['medical_condition']}\n"
            f"   Drug Class : {row['drug_classes']}\n"
            f"   Brand Names: {row['brand_names']}\n"
            f"   Rx / OTC   : {row['rx_otc']}\n"
            f"   Rating     : {row['rating']}/10\n\n"
            "⚠️ Consult a doctor before taking any medicine."
        )
    except Exception as e:
        return f"⚠️ Error fetching medicine info: {e}"


@tool
def check_drug_interactions(drugs: str) -> str:
    """Lists individual warnings (alcohol, pregnancy, Rx status) for two or more medicines.
    Note: cross-drug interaction data is not available in this database.
    Input should be medicine names separated by commas."""
    if DF.empty:
        return "⚠️ Medicine data unavailable. Please ensure the CSV file is correctly placed."
    try:
        if not drugs or not drugs.strip():
            return "⚠️ Please provide medicine names separated by commas to check for interactions."
        drug_list = [d.strip() for d in drugs.split(",") if d.strip()]
        
        if len(drug_list)<2:
            return "⚠️ Please provide at least two medicine names separated by commas to check for interactions."
        if len(drug_list) > 5:
            return "⚠️ Please check a maximum of 5 medicines at a time."

        results = []
        not_found = []
        for drug in drug_list:
            rows = _find_rows(drug)
            if rows.empty:
                not_found.append(drug)
            else:
                results.append(rows.iloc[0])
        if not results:
            return f"❌ None of the medicines ({', '.join(drug_list)}) were found. Try using generic names."
        response = f"🔍 Interaction check for: {', '.join(drug_list)}\n\n"
        for rows in results:
            response += f"💊 {rows['drug_name']} (Rx/OTC: {rows['rx_otc']})\n"
            if str(rows['alcohol']).strip().upper() == "X":
                response += "  ⚠️ Avoid alcohol with this medicine.\n"
            if rows['rx_otc'] == "Rx":
                response += "  🔒 Prescription only.\n"
            if str(rows['pregnancy_category']).strip().upper() in ("D", "X"):
                response += f"  🤰 Pregnancy risk: Category {rows['pregnancy_category']}.\n"
        if not_found:
            response += f"\n⚠️ Not found in database: {', '.join(not_found)}\n"
        for drug_row in results:                # FIX #4: renamed loop var from 'rows' to 'drug_row'
            response += f"💊 {drug_row['drug_name']} (Rx/OTC: {drug_row['rx_otc']})\n"
            if str(drug_row['alcohol']).strip().upper() == "X":
                response += "  ⚠️ Avoid alcohol with this medicine.\n"
            if drug_row['rx_otc'] == "Rx":
                response += "  🔒 Prescription only.\n"
            if str(drug_row['pregnancy_category']).strip().upper() in ("D", "X"):
                response += f"  🤰 Pregnancy risk: Category {drug_row['pregnancy_category']}.\n"

        drug_classes = []
        for drug_row in results:
            cls = str(drug_row.get('drug_classes', '')).strip().lower()
            if cls and cls != 'nan':
                drug_classes.append((drug_row['drug_name'], cls))

        seen_classes = {}
        for name, cls in drug_classes:
            if cls in seen_classes:
                response += (
                    f"\n⚠️ WARNING: '{name}' and '{seen_classes[cls]}' are in the same drug class "
                    f"({cls}). Taking both may increase the risk of side effects.\n"
                )
            else:
                seen_classes[cls] = name

        if not_found:
            response += f"\n⚠️ Not found in database: {', '.join(not_found)}\n"
        
        response += (
            "\n⚠️ IMPORTANT: This tool can only show individual drug warnings.\n"
            "It cannot detect all drug-drug interactions.\n"
            "For a full interaction check, consult a pharmacist or visit drugs.com/interactions."
        )
        return response
    except Exception as e:
        return f"⚠️ Error checking drug interactions: {e}"

@tool
def suggest_medicine_for_symptoms(symptom: str) -> str:
    """Suggest commonly used medicine for a symptom or disease or condition. Always remind the user to consult 
    a doctor or specialist before taking any medicine.  Use when user describes symptoms and asks what medicine
    to take."""
    if DF.empty:
        return "⚠️ Symptom data unavailable. Please ensure the CSV file is correctly placed."
    try:
        if not symptom or not symptom.strip():
            return "⚠️ Please provide valid symptoms or condition to get medicine suggestions."
        rows = _find_condition(symptom)
        if rows.empty:
            conditions = DF["medical_condition"].dropna().unique()[:15]
            return (
                f"❌ No results for '{symptom}'.\n"
                f"Available conditions: {', '.join(conditions)}"
            )
        drugs = rows["drug_name"].dropna().unique()[:5]
        return (
            f"for symptom/condition '{symptom}', commonly used medicines include:\n"
            f"{', '.join(drugs)}\n\n"
            f"⚠️ Always consult a doctor before taking any medicine."
        )
    except Exception as e:
        return f"⚠️ Error suggesting medicine for symptoms: {e}"
@tool
def get_side_effects(medicine: str) -> str:
    """Return common side effects and precautions for a given medicine. Use when user asks about side effects of a specific medicine."""
    if DF.empty:
        return "⚠️ Medicine data unavailable. Please ensure the CSV file is correctly placed."
    try:
        if not medicine or not medicine.strip():
            return "⚠️ Please provide a valid medicine name to get side effect information."
        rows = _find_rows(medicine)
        if rows.empty:
            return f"⚠️ No information found for medicine: '{medicine}'. Please check the name and try again."
        row = rows.iloc[0]
        side_effects = str(row['side_effects'])
        return (
            f"⚠️ Side Effects for: {row['drug_name']}\n\n"
            f"{side_effects}\n\n"
            f"🤰Pregnancy Precautions: {_pregnancy_label(str(row['pregnancy_category']))}\n"
            f"🍺 Alcohol Warning    : {'⚠️ Avoid alcohol' if str(row['alcohol']).strip().upper() == 'X' else 'No major interaction noted'}\n\n"
            "If you experience severe side effects, consult a doctor immediately."
        )
    except Exception as e:
        return f"⚠️ Error fetching side effect information: {e}"


@tool
def dosage_guide(medicine: str) -> str:
    """Return standard dosage information for a given medicine. Always remind the user to consult a doctor or specialist
     for personalized dosage recommendations. Use when user asks about dosage information for a specific medicine."""
    if DF.empty:
        return "⚠️ Medicine data unavailable. Please ensure the CSV file is correctly placed."
    try:
        if not medicine or not medicine.strip():
            return "⚠️ Please provide a valid medicine name to get dosage information."
        rows = _find_rows(medicine)
        if rows.empty:
            return f"⚠️ No information found for medicine: '{medicine}'. Please check the name and try again."
        row = rows.iloc[0]
        return (
            f"⚠️ Dosage Guide for: {row['drug_name']}\n\n"
            f"   Generic Name : {row['generic_name']}\n"
            f"   Drug Class   : {row['drug_classes']}\n"
            f"   Rx / OTC     : {row['rx_otc']}\n\n"
            "📋 Note: Specific dosage amounts are not available in our database.\n"
            "   Dosage varies by age, weight, and condition severity.\n\n"
            "✅ Consult your doctor or pharmacist for exact dosage recommendations."
        )
    except Exception as e:
        return f"⚠️ Error fetching dosage information: {e}"

@tool
def escalate_to_doctor(symptoms: str) -> str:
    """ONLY use this tool if:

- chest pain
- unconsciousness
- severe bleeding
- stroke symptoms
- suicidal thoughts
- overdose
- difficulty breathing
- life-threatening emergency

DO NOT use this tool for:
- medicine information
- side effects
- dosage
- drug interactions
- pregnancy category
- alcohol warning
"""
    return f"⚠️ This query requires professional medical attention. Please consult a registered doctor. Reason: {symptoms}"


def create_agent() -> AgentExecutor:
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=2048,
        timeout=60,
        max_retries=2,
    )

    tools = [
        get_medicine_info,
        check_drug_interactions,
        suggest_medicine_for_symptoms,
        get_side_effects,
        dosage_guide,
        escalate_to_doctor
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are MedAssist, a medical information assistant.

        STRICT RULES — NEVER BREAK THESE:
        - Call exactly ONE tool unless the user's question genuinely requires multiple tools.

        After receiving the tool output,
        answer using ONLY that output.

        Do not replace the tool output with the result of another tool.

        Only call escalate_to_doctor for genuine medical emergencies.
        - NEVER use your own knowledge to answer medical questions.
        - NEVER guess or make up medicine information.
        - If a tool returns '❌ not found in our database' — tell the user exactly that.
        Do NOT fill in with information from your training data.
        - Only provide information that comes directly from tool results.
        - Always remind users responses are for informational purposes only.

        TOOL USAGE:
        - Side effects question       → call get_side_effects
        - Medicine info question      → call get_medicine_info  
        - Symptom/condition question  → call suggest_medicine_for_symptoms
        - Dosage question             → call dosage_guide
        - Interaction question        → call check_drug_interactions
        - Emergency/serious symptoms  → call escalate_to_doctor
        """),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent = agent,
        tools = tools,
        verbose = False,
        handle_parsing_errors = True,
        max_iterations=10,                
        return_intermediate_steps=False,
    )
    return agent_executor

def chat(user_input: str, agent_executor: AgentExecutor) -> str:
    global chat_history
    try:
        if chat_history is None:
            chat_history = []
        if agent_executor is None:
            agent_executor = create_agent()
        try:
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": list(chat_history),
            })
            if response is None:
                output = "No response generated. Please try again."
            elif isinstance(response, dict):
                output = response.get("output", "") or "Could not generate a response. Please rephrase."
            elif hasattr(response, "output"):
                output = str(response.output)
            else:
                output = str(response)
        except Exception as e:
            output = f"I encountered an error: {str(e)}. Please rephrase your question."
        if not output:
            output = "I'm not sure how to respond. Could you rephrase?"
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=output))
        chat_history = chat_history[-20:]

        return output

    except Exception as e:
        print(f"Error in chat function: {e}")
        return "I encountered an error. Please try again."

def main():
    agent_executor = create_agent()
    print("🏥 MedAssist is ready! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if not user_input.strip():
            print("Please enter a valid message.")
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye! Stay healthy!")
            break
        response = chat(user_input, agent_executor)
        print(f"MedAssist: {response}\n")

if __name__ == "__main__":
    main()