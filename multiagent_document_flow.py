from typing import Annotated, Literal, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from pydantic import BaseModel, Field
from llm_instances import default_openai_model, default_mistral_model, ModelConfig, create_model

# Define the state schema for our multi-agent system
class HealthcareState(BaseModel):
    """State for the healthcare multi-agent workflow."""
    messages: list = Field(default_factory=list)
    patient_id: str = ""
    document_type: str = ""
    final_output: dict = Field(default_factory=dict)
    related_documents: list = Field(default_factory=list)

def create_healthcare_workflow(model_config: ModelConfig = None):
    """Create a healthcare workflow with the specified model configuration."""
    # Initialize the model - use default OpenAI model if no config provided
    model = create_model(model_config) if model_config else default_openai_model
    
    def router(state: HealthcareState) -> Command[Literal["document_agent", "medicine_agent", "medical_gadget_agent"]]:
        """Router agent that decides which specialized agent to call next."""
        router_prompt = SystemMessage(content="""
        You are a healthcare workflow router. Based on the patient ID and context, decide which specialized 
        agent should handle this case:
        - document_agent: For document processing, records management
        - medicine_agent: For medication-related queries and records
        - medical_gadget_agent: For medical devices and equipment
        
        Return your decision as the name of the agent only.
        """)
        
        history = state.messages.copy()
        current_input = HumanMessage(content=f"Patient ID: {state.patient_id}. Please route to the appropriate agent.")
        
        response = model.invoke([router_prompt] + history + [current_input])
        
        agent_decision = response.content.strip().lower()
        if "document" in agent_decision:
            next_agent = "document_agent"
        elif "medicine" in agent_decision:
            next_agent = "medicine_agent"
        elif "gadget" in agent_decision or "device" in agent_decision or "equipment" in agent_decision:
            next_agent = "medical_gadget_agent"
        else:
            next_agent = "document_agent"
        
        router_message = AIMessage(content=f"Routing to {next_agent}")
        
        return Command(
            goto=next_agent,
            update={"messages": state.messages + [current_input, router_message],
                    "document_type": next_agent.split("_")[0]}
        )

    def document_agent(state: HealthcareState) -> Command[Literal["document_relationship_agent"]]:
        """Agent specialized in document processing."""
        document_prompt = SystemMessage(content="""
        You are a healthcare document specialist. Process the patient's document requests and provide
        comprehensive documentation. Focus on records management, patient history documentation,
        and other document-related tasks.
        """)
        
        current_task = HumanMessage(content=f"Process document for Patient ID: {state.patient_id}. Document type: {state.document_type}")
        
        response = model.invoke([document_prompt] + state.messages + [current_task])
        
        document_output = {
            "id": f"doc-{state.patient_id}",
            "patient_id": state.patient_id,
            "document_description": response.content,
            "type": "document"
        }
        
        return Command(
            goto="document_relationship_agent",
            update={
                "messages": state.messages + [current_task, response],
                "final_output": document_output
            }
        )

    def document_relationship_agent(state: HealthcareState) -> Command[Literal["finalize_output"]]:
        """Agent that checks for relationships between documents in the database."""
        relationship_prompt = SystemMessage(content="""
        You are a healthcare document relationship specialist. Analyze the current document and
        check for any relationships with other documents in the patient's history. Look for
        dependencies, references, or related documentation that might be relevant.
        """)
        
        # Placeholder for database retrieval
        def retrieve_stored_data(patient_id: str) -> list:
            """Placeholder function for retrieving data from Supabase database."""
            # In a real implementation, this would query Supabase
            return [
                {"id": "doc-123", "type": "prescription", "date": "2024-01-01"},
                {"id": "doc-456", "type": "lab_result", "date": "2024-01-15"}
            ]
        
        # Retrieve related documents
        related_docs = retrieve_stored_data(state.patient_id)
        
        current_task = HumanMessage(content=f"""
        Analyze relationships for Patient ID: {state.patient_id}
        Current document: {state.final_output}
        Related documents: {related_docs}
        """)
        
        response = model.invoke([relationship_prompt] + state.messages + [current_task])
        
        relationship_output = {
            **state.final_output,
            "related_documents": related_docs,
            "relationship_analysis": response.content
        }
        
        return Command(
            goto="finalize_output",
            update={
                "messages": state.messages + [current_task, response],
                "final_output": relationship_output,
                "related_documents": related_docs
            }
        )

    def medicine_agent(state: HealthcareState) -> Command[Literal["finalize_output"]]:
        """Agent specialized in medicine and pharmacy."""
        medicine_prompt = SystemMessage(content="""
        You are a healthcare medicine specialist. Handle medication-related tasks including
        prescriptions, medication reviews, and pharmaceutical documentation for patients.
        """)
        
        current_task = HumanMessage(content=f"Process medication information for Patient ID: {state.patient_id}")
        
        response = model.invoke([medicine_prompt] + state.messages + [current_task])
        
        medicine_output = {
            "id": f"med-{state.patient_id}",
            "patient_id": state.patient_id,
            "medication_description": response.content,
            "type": "medication"
        }
        
        return Command(
            goto="finalize_output",
            update={
                "messages": state.messages + [current_task, response],
                "final_output": medicine_output
            }
        )

    def medical_gadget_agent(state: HealthcareState) -> Command[Literal["finalize_output"]]:
        """Agent specialized in medical devices and equipment."""
        gadget_prompt = SystemMessage(content="""
        You are a healthcare medical device specialist. Handle tasks related to medical equipment,
        devices, implants, and other medical technology for patients.
        """)
        
        current_task = HumanMessage(content=f"Process medical device information for Patient ID: {state.patient_id}")
        
        response = model.invoke([gadget_prompt] + state.messages + [current_task])
        
        gadget_output = {
            "id": f"device-{state.patient_id}",
            "patient_id": state.patient_id,
            "device_description": response.content,
            "type": "medical_device"
        }
        
        return Command(
            goto="finalize_output",
            update={
                "messages": state.messages + [current_task, response],
                "final_output": gadget_output
            }
        )

    def finalize_output_node(state: HealthcareState) -> Dict[str, Any]:
        """Final node that formats and returns the completed output."""
        final_prompt = SystemMessage(content="""
        You are a healthcare documentation finalizer. Create a final, comprehensive document
        that includes all relevant information from the processing that occurred.
        """)
        
        summary_task = HumanMessage(content=f"""
        Create final output document for Patient ID: {state.patient_id}
        Document type: {state.document_type}
        Content to include: {state.final_output}
        """)
        
        response = model.invoke([final_prompt] + [summary_task])
        
        final_document = {
            **state.final_output,
            "finalized_text": response.content,
            "completed": True
        }
        
        return {"messages": state.messages + [summary_task, response], 
                "final_output": final_document}

    # Create the StateGraph
    workflow_builder = StateGraph(HealthcareState)
    
    # Add all our nodes
    workflow_builder.add_node("router", router)
    workflow_builder.add_node("document_agent", document_agent)
    workflow_builder.add_node("document_relationship_agent", document_relationship_agent)
    workflow_builder.add_node("medicine_agent", medicine_agent)
    workflow_builder.add_node("medical_gadget_agent", medical_gadget_agent)
    workflow_builder.add_node("finalize_output", finalize_output_node)
    
    # Define the edges
    workflow_builder.add_edge(START, "router")
    workflow_builder.add_edge("document_agent", "document_relationship_agent")
    workflow_builder.add_edge("document_relationship_agent", "finalize_output")
    workflow_builder.add_edge("medicine_agent", "finalize_output")
    workflow_builder.add_edge("medical_gadget_agent", "finalize_output")
    workflow_builder.add_edge("finalize_output", END)
    
    # Compile the graph
    return workflow_builder.compile()

def process_patient_request(patient_id: str, model_config: ModelConfig = None):
    """Process a patient request using the specified model configuration."""
    workflow = create_healthcare_workflow(model_config)
    initial_state = HealthcareState(patient_id=patient_id)
    result = workflow.invoke(initial_state)
    return result.final_output

# Example usage
if __name__ == "__main__":
    from llm_instances import ModelProvider, ModelConfig
    
    # Example configurations
    openai_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4-turbo",
        temperature=0.0
    )
    
    mistral_config = ModelConfig(
        provider=ModelProvider.MISTRAL,
        model_name="mistral-large-latest",
        temperature=0.0
    )
    
    google_config = ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_name="gemini-2.0-flash",
        temperature=0.0
    )
    
    # # Process a request using OpenAI
    # openai_result = process_patient_request("P12345", openai_config)
    # print(f"OpenAI Result: {openai_result}")
    
    # Process a request using Mistral
    mistral_result = process_patient_request("P12345", google_config)
    print(f"Mistral Result: {mistral_result}")
    
    # # Process a request using default model (OpenAI)
    # default_result = process_patient_request("P12345")
    # print(f"Default Result: {default_result}")