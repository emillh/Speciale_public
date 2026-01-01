import operator
from typing import Annotated, List, TypedDict
import time

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

try:
    from .report_schema import LeadReport
except ImportError:
    from report_schema import LeadReport

class ConversationState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    report: LeadReport
    turn_number: int

# --- Agent Node Functions ---

def ask_question_node(state: ConversationState, llm):
    """Asks a question based on the current turn to gather information for a specific section."""
    print(f"--- Sales Agent Node (Turn {state['turn_number']}) ---")
    
    sections = [
        "Basic Company and Contact Information",
        "Goals & Motivation",
        "Technical Details",
        "Commercial Information",
        "General Company Background"
    ]
    turn = state['turn_number']
    # Cycle through the sections using the modulo operator
    current_section = sections[turn % len(sections)]
    
    # --- PROMPT IMPROVEMENT ---
    prompt_messages = [
        ("system", 
         f"""You are a world-class sales agent conducting a lead qualification call.
         
Your Goal: Gather information about the topic: **{current_section}**.

Rules:
1.  **Acknowledge and Transition:** Briefly acknowledge the customer's previous answer before asking your next question to make the conversation feel natural.
2.  **Be Strategic:** Look at the 'Current Report' below. Do not ask for information that is already filled out. Instead, ask a question that targets missing information within the current topic.
3.  **Be Clear and Concise:** Ask only one, clear question. Do not mention the report or section names.

Current Report:
```json
{state['report'].model_dump_json(exclude_unset=True)}
```

Conversation History:
"""),
    ]
    prompt_messages.extend(state['messages'])
    prompt_messages.append(("human", "Please ask your next strategic question."))
    # --- END OF IMPROVEMENT ---

    response = llm.invoke(prompt_messages) 
    question = response.content
    
    print(f"Sales Agent asking: {question}")
    time.sleep(5) # Add a delay to manage rate limits
    
    return {"messages": [AIMessage(content=question, name="SalesAgent")], "turn_number": state["turn_number"] + 1}

def customer_node(state: ConversationState, llm, persona_report: LeadReport):
    """Simulates the customer's response based on a persona defined in a LeadReport object."""
    print("--- Customer Node ---")
    
    # --- PROMPT IMPROVEMENT ---
    system_prompt = f"""You are acting as a person whose details are defined in the following JSON object. 
Your name is Alex, and you are a Project Manager.
Answer the sales agent's questions naturally, using only the information provided in this JSON. 

Rules:
1.  **Stay in Character:** Do not mention that you are an AI or that you are using a JSON object.
2.  **Use Only Your Persona:** Base all your answers on the "Your Persona Details" provided below.
3.  **Handle Missing Information:** If the agent asks for information not present in your persona, politely state that you don't have that detail.

**Your Persona Details:**
```json
{persona_report.model_dump_json()}
```
"""
    # --- END OF IMPROVEMENT ---
    
    prompt_messages = [
        ("system", system_prompt),
        *state["messages"]
    ]
    
    response = llm.invoke(prompt_messages)
    response_text = response.content
    
    print(f"Customer generated: {response_text}")
    time.sleep(2) # Add a delay to manage rate limits
    return {"messages": [HumanMessage(content=response_text, name="Customer")]}

def extract_info_node(state: ConversationState, llm):
    """Extracts information from the conversation and updates the report using structured output."""
    print("--- Extract Info Node ---")
    if not any(isinstance(msg, HumanMessage) for msg in state['messages']):
        return {}

    structured_llm = llm.with_structured_output(LeadReport)
    
    # --- PROMPT IMPROVEMENT ---
    # Focus the prompt on the last exchange for better accuracy
    last_sales_agent_message = state['messages'][-2].content
    last_customer_message = state['messages'][-1].content

    prompt_messages = [
        ("system", """You are a data entry specialist. Your task is to update a JSON report based on the last turn of a conversation.
        You will be given the current report and the last question/answer from the conversation.
        Update the report with any new information from the customer's answer.
        Only fill in fields for which you have explicit information. Do not guess or infer values.
        Return the complete, updated JSON object, carrying over all existing data."""),
        ("human", f"""Current Report:
        {state['report'].model_dump_json()}
        
        Last Conversation Turn:
        - Sales Agent asked: "{last_sales_agent_message}"
        - Customer answered: "{last_customer_message}"
        
        Based on the customer's answer, provide the updated report object.""")
    ]
    # --- END OF IMPROVEMENT ---

    try:
        updated_report = structured_llm.invoke(prompt_messages)
        print(f"Updated Report: {updated_report.model_dump_json(indent=2)}")
        time.sleep(5) # Add a delay to manage rate limits
        return {"report": updated_report}
    except Exception as e:
        print(f"--- Error extracting info: {e} ---")
        return {"report": state["report"]}

# --- Edge Logic ---
MAX_TURNS = 5
def should_continue(state: ConversationState):
    if state["turn_number"] >= MAX_TURNS:
        return "end"
    else:
        return "continue"

# --- Build the Graph ---
def build_graph(llm_conversation, llm_extractor, customer_persona_report: LeadReport):
    """
    Builds the LangGraph StateGraph for the lead qualification bot.
    This version uses different LLMs for conversation and extraction.
    """
    workflow = StateGraph(ConversationState)

    # Define nodes, passing the appropriate LLM to each
    workflow.add_node("sales_agent", lambda state: ask_question_node(state, llm_conversation))
    workflow.add_node("customer", lambda state: customer_node(state, llm_conversation, customer_persona_report))
    workflow.add_node("extractor", lambda state: extract_info_node(state, llm_extractor))

    # Define graph flow
    workflow.set_entry_point("sales_agent")
    workflow.add_edge("sales_agent", "customer")
    workflow.add_edge("customer", "extractor")
    
    workflow.add_conditional_edges(
        "extractor",
        should_continue,
        {"continue": "sales_agent", "end": END}
    )

    app = workflow.compile()
    print("--- Graph Compiled for New Generation ---")
    return app