from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import json
import pandas as pd

load_dotenv()
CSV_PATH = os.path.join(os.path.dirname(__file__), "drugs_side_effects_drugs_com.csv")

# Single knob to control "how many results" everywhere in this file.
TOP_N = 3

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
 
def _top_side_effects(raw_text: str, n: int = TOP_N) -> list[str]:
    """
    The CSV's side_effects field is free-form prose (serious warnings +
    a 'Common side effects ... may include: a; b; c.' tail), not a clean
    list. This is a best-effort heuristic, not guaranteed-perfect parsing:
      1. Prefer the text after 'may include:' (usually the common/mild list).
      2. Fall back to the whole field if that phrase isn't present.
      3. Split on ';' (the CSV's usual separator); if that yields only one
         piece, try splitting on '. ' instead.
      4. Clean up fragments and return the first n non-trivial ones.
    """
    if not raw_text or str(raw_text).strip().lower() == "nan":
        return []
    text = str(raw_text)
    marker = "may include:"
    idx = text.lower().rfind(marker)
    segment = text[idx + len(marker):] if idx != -1 else text

    parts = [p.strip(" .;") for p in segment.split(";")]
    if len(parts) < 2:
        parts = [p.strip(" .;") for p in segment.split(". ")]

    cleaned = [p for p in parts if p and len(p) > 2]
    return cleaned[:n]


def _clean(value):
    """Convert pandas NaN / NaT / numpy types into JSON-safe native Python values.
    Raw NaN is not valid JSON and will break strict parsers downstream."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):  # numpy int64/float64 -> native python
        return value.item()
    return value

def _to_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)

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
    """Returns uses, indications, dosage, and drug class for a given medicine name, as a JSON string.
    Use this when the user asks what a specific medicine is used for."""
    if DF.empty:
        return _to_json({"error": "Medicine information is currently unavailable. Please ensure the CSV file is correctly placed."})
    try:
        if not medicine_name or not medicine_name.strip():
            return _to_json({"error": "Please provide a valid medicine name."})
        rows = _find_rows(medicine_name)
        if rows.empty:
            return _to_json({"error": f"No information found for medicine: '{medicine_name}'. Please check the name and try again."})
        row = rows.iloc[0]
        return _to_json({
            "drug_name": _clean(row["drug_name"]),
            "used_for": _clean(row["medical_condition"]),
            "drug_class": _clean(row["drug_classes"]),
            "brand_names": _clean(row["brand_names"]),
            "rx_otc": _clean(row["rx_otc"]),
            "rating": _clean(row["rating"]),
            "disclaimer": "Consult a doctor before taking any medicine."
        })
    except Exception as e:
        return _to_json({"error": f"Error fetching medicine info: {e}"})


@tool
def check_drug_interactions(drugs: str) -> str:
    """Lists individual warnings (alcohol, pregnancy, Rx status) for two or more medicines, as a JSON string.
    Note: cross-drug interaction data is not available in this database.
    Input should be medicine names separated by commas."""
    if DF.empty:
        return _to_json({"error": "Medicine data unavailable. Please ensure the CSV file is correctly placed."})
    try:
        if not drugs or not drugs.strip():
            return _to_json({"error": "Please provide medicine names separated by commas to check for interactions."})
        drug_list = [d.strip() for d in drugs.split(",") if d.strip()]

        if len(drug_list) < 2:
            return _to_json({"error": "Please provide at least two medicine names separated by commas to check for interactions."})
        if len(drug_list) > 5:
            return _to_json({"error": "Please check a maximum of 5 medicines at a time."})

        results = []
        not_found = []
        for drug in drug_list:
            rows = _find_rows(drug)
            if rows.empty:
                not_found.append(drug)
            else:
                results.append(rows.iloc[0])
        if not results:
            return _to_json({"error": f"None of the medicines ({', '.join(drug_list)}) were found. Try using generic names."})

        # Build a flat list of individual warning objects (one loop, no duplication),
        # then only keep the top N overall.
        warnings = []
        for drug_row in results:
            name = drug_row['drug_name']
            if str(drug_row['alcohol']).strip().upper() == "X":
                warnings.append({"drug": _clean(name), "type": "alcohol", "message": "Avoid alcohol with this medicine."})
            if drug_row['rx_otc'] == "Rx":
                warnings.append({"drug": _clean(name), "type": "rx_only", "message": "Prescription only."})
            if str(drug_row['pregnancy_category']).strip().upper() in ("D", "X"):
                warnings.append({"drug": _clean(name), "type": "pregnancy", "message": f"Pregnancy risk: Category {drug_row['pregnancy_category']}."})

        drug_classes = []
        for drug_row in results:
            cls = str(drug_row.get('drug_classes', '')).strip().lower()
            if cls and cls != 'nan':
                drug_classes.append((drug_row['drug_name'], cls))

        same_class_warnings = []
        seen_classes = {}
        for name, cls in drug_classes:
            if cls in seen_classes:
                same_class_warnings.append({
                    "drug_a": _clean(name),
                    "drug_b": _clean(seen_classes[cls]),
                    "drug_class": cls,
                    "message": "Same drug class — taking both may increase the risk of side effects."
                })
            else:
                seen_classes[cls] = name
        warnings.extend(same_class_warnings)

        top_warnings = warnings[:TOP_N]

        return _to_json({
            "drugs_checked": drug_list,
            "not_found": not_found,
            "warnings": top_warnings,
            "total_warnings_found": len(warnings),
            "note": "This tool can only show individual drug warnings, not all drug-drug interactions. For a full interaction check, consult a pharmacist or visit drugs.com/interactions.",
            "disclaimer": "Always consult a doctor before taking any medicine."
        })
    except Exception as e:
        return _to_json({"error": f"Error checking drug interactions: {e}"})

@tool
def suggest_medicine_for_symptoms(symptom: str) -> str:
    """Suggest commonly used medicine for a symptom or disease or condition, as a JSON string. Always remind the
    user to consult a doctor or specialist before taking any medicine. Use when user describes symptoms and asks
    what medicine to take."""
    if DF.empty:
        return _to_json({"error": "Symptom data unavailable. Please ensure the CSV file is correctly placed."})
    try:
        if not symptom or not symptom.strip():
            return _to_json({"error": "Please provide valid symptoms or condition to get medicine suggestions."})
        rows = _find_condition(symptom)
        if rows.empty:
            conditions = DF["medical_condition"].dropna().unique()[:TOP_N].tolist()
            return _to_json({
                "error": f"No results for '{symptom}'.",
                "closest_available_conditions": conditions
            })
        # Prefer higher-rated medicines when picking the top N, same logic as _find_rows.
        subset = rows.dropna(subset=["drug_name"])
        if "rating" in subset.columns:
            subset = subset.sort_values("rating", ascending=False)
        drugs = subset["drug_name"].drop_duplicates().head(TOP_N).tolist()
        return _to_json({
            "symptom": symptom,
            "suggested_medicines": drugs,
            "disclaimer": "Always consult a doctor before taking any medicine."
        })
    except Exception as e:
        return _to_json({"error": f"Error suggesting medicine for symptoms: {e}"})
@tool
def get_side_effects(medicine: str) -> str:
    """Return common side effects and precautions for a given medicine, as a JSON string. Use when user asks
    about side effects of a specific medicine."""
    if DF.empty:
        return _to_json({"error": "Medicine data unavailable. Please ensure the CSV file is correctly placed."})
    try:
        if not medicine or not medicine.strip():
            return _to_json({"error": "Please provide a valid medicine name to get side effect information."})
        rows = _find_rows(medicine)
        if rows.empty:
            return _to_json({"error": f"No information found for medicine: '{medicine}'. Please check the name and try again."})
        row = rows.iloc[0]
        top_effects = _top_side_effects(row['side_effects'], TOP_N)
        return _to_json({
            "drug_name": _clean(row["drug_name"]),
            "side_effects": top_effects,
            "pregnancy_category": _pregnancy_label(str(row['pregnancy_category'])),
            "alcohol_warning": "Avoid alcohol" if str(row['alcohol']).strip().upper() == 'X' else "No major interaction noted",
            "note": "If you experience severe side effects, consult a doctor immediately."
        })
    except Exception as e:
        return _to_json({"error": f"Error fetching side effect information: {e}"})


@tool
def dosage_guide(medicine: str) -> str:
    """Return standard dosage information for a given medicine, as a JSON string. Always remind the user to
    consult a doctor or specialist for personalized dosage recommendations. Use when user asks about dosage
    information for a specific medicine."""
    if DF.empty:
        return _to_json({"error": "Medicine data unavailable. Please ensure the CSV file is correctly placed."})
    try:
        if not medicine or not medicine.strip():
            return _to_json({"error": "Please provide a valid medicine name to get dosage information."})
        rows = _find_rows(medicine)
        if rows.empty:
            return _to_json({"error": f"No information found for medicine: '{medicine}'. Please check the name and try again."})
        row = rows.iloc[0]
        return _to_json({
            "drug_name": _clean(row["drug_name"]),
            "generic_name": _clean(row["generic_name"]),
            "drug_class": _clean(row["drug_classes"]),
            "rx_otc": _clean(row["rx_otc"]),
            "note": "Specific dosage amounts are not available in our database. Dosage varies by age, weight, and condition severity.",
            "disclaimer": "Consult your doctor or pharmacist for exact dosage recommendations."
        })
    except Exception as e:
        return _to_json({"error": f"Error fetching dosage information: {e}"})

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
    return _to_json({
        "escalation": True,
        "reason": symptoms,
        "message": "This query requires professional medical attention. Please consult a registered doctor."
    })


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

        Tool outputs are JSON strings holding structured data (or an "error" key).
        After receiving the tool output, translate that JSON into a clear, natural-language
        answer for the user — do not print raw JSON to the user, and answer using ONLY
        the fields present in that JSON.

        Do not replace the tool output with the result of another tool.

        Only call escalate_to_doctor for genuine medical emergencies.
        - NEVER use your own knowledge to answer medical questions.
        - NEVER guess or make up medicine information.
        - If a tool's JSON contains an "error" key — tell the user exactly that message.
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
        return_intermediate_steps=True,
    )
    return agent_executor

def chat(user_input: str, agent_executor: AgentExecutor, session_history: list = None) -> dict:
    """
    session_history: pass a list of HumanMessage/AIMessage for THIS conversation only.
    Pass None to fall back to the module-level global (only safe for a single-user CLI —
    NEVER pass None in a multi-user server, or every user's history will bleed together).

    Returns a dict:
      {
        "reply": str,            # natural-language answer for the user
        "tool_results": list,    # raw structured data from each tool call this turn, e.g.
                                  # [{"tool": "get_side_effects", "input": {...}, "data": {...}}]
        "history": list          # updated history — caller must store this and pass it back in next turn
      }
    tool_results holds the actual parsed JSON each tool returned (already json.loads'd),
    independent of how the LLM chose to phrase its reply — use this for anything programmatic.
    """
    global chat_history
    using_global = session_history is None
    if using_global:
        if chat_history is None:
            chat_history = []
        history = chat_history
    else:
        history = session_history

    tool_results = []
    try:
        if agent_executor is None:
            agent_executor = create_agent()
        try:
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": list(history),
            })
            if response is None:
                output = "No response generated. Please try again."
            elif isinstance(response, dict):
                output = response.get("output", "") or "Could not generate a response. Please rephrase."
                for action, tool_output in response.get("intermediate_steps", []):
                    try:
                        parsed = json.loads(tool_output)
                    except (TypeError, json.JSONDecodeError):
                        parsed = tool_output  # tool didn't return JSON (shouldn't happen, but stay safe)
                    tool_results.append({
                        "tool": action.tool,
                        "input": action.tool_input,
                        "data": parsed,
                    })
            elif hasattr(response, "output"):
                output = str(response.output)
            else:
                output = str(response)
        except Exception as e:
            output = f"I encountered an error: {str(e)}. Please rephrase your question."
        if not output:
            output = "I'm not sure how to respond. Could you rephrase?"
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=output))
        history = history[-20:]
        if using_global:
            chat_history = history

        return {"reply": output, "tool_results": tool_results, "history": history}

    except Exception as e:
        print(f"Error in chat function: {e}")
        return {"reply": "I encountered an error. Please try again.", "tool_results": tool_results, "history": history}

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
        result = chat(user_input, agent_executor)
        if result["tool_results"]:
            for tr in result["tool_results"]:
                print(json.dumps(tr["data"], indent=2, ensure_ascii=False))
        else:
            # No tool was called this turn (e.g. small talk) — there's no structured
            # data to show, so fall back to the plain reply so the loop isn't silent.
            print(json.dumps({"reply": result["reply"]}, ensure_ascii=False))
        print()

if __name__ == "__main__":
    main()
