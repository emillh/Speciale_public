from pydantic import BaseModel, Field
from typing import Optional, List

class LeadReport(BaseModel):
    """
    Represents the structured report to be filled by the Interviewer Agent.
    This schema is based on a 30-question form for lead qualification.
    """

    # Section 1: Basic Company and Contact Information
    company_name: Optional[str] = Field(None, description="Official name of the potential client's company.")
    contact_person_name: Optional[str] = Field(None, description="Full name of the primary contact person.")
    contact_person_role: Optional[str] = Field(None, description="Role or title of the contact person within their company.")
    contact_person_email: Optional[str] = Field(None, description="Email address of the contact person.")
    contact_person_phone: Optional[str] = Field(None, description="Phone number of the contact person.")
    is_decision_maker: Optional[bool] = Field(None, description="Is the contact person the final decision-maker for this potential project/purchase?")
    industry: Optional[str] = Field(None, description="The primary industry the company operates in. (Corresponds to survey 1.1.2)")
    address: Optional[str] = Field(None, description="Full physical address of the company. (Corresponds to survey 1.1.3)")


    # Section 2: Goals & Motivation (Corresponds to survey section 2)
    purpose_of_request: Optional[str] = Field(None, description="The client's stated purpose for this request or for considering automation. (Corresponds to survey 2.1.1)")
    # Free text inspiration questions from survey section 2 (Why/Automation, Pain Points, How did you hear about UR) can be summarized or key themes extracted by the LLM into broader fields if desired, or left as conversational context.
    # For direct Pydantic fields, we'll map the numbered/specific ones:
    lead_source: Optional[str] = Field(None, description="How the client learned about our company/offerings. (Corresponds to survey 'Lead source')")
    worked_with_ur_before: Optional[str] = Field(None, description="Has the client worked with UR before? (Yes/No or details). (Corresponds to survey 2.1.2)")
    in_house_automation_resources_description: Optional[str] = Field(None, description="Description of existing automation technologies or systems the client currently uses.")
    in_house_team_skill_level: Optional[str] = Field(None, description="Skill level of the client's team in managing automation systems (e.g., beginner, intermediate, expert).")
    installation_responsibility: Optional[str] = Field(None, description="Who will be responsible for overseeing the installation process? In-house or external? (Corresponds to survey 2.1.3)")
    preferred_installation_timeline: Optional[str] = Field(None, description="Client's preferred timeline for completing the installation (e.g., specific date, timeframe). (Corresponds to survey 2.1.4)")
    deployment_process_preference: Optional[str] = Field(None, description="Client's preferred deployment process. (Corresponds to survey 2.1.5)")


    # Section 3: Technical Details (Corresponds to survey section 3)
    application_description: Optional[str] = Field(None, description="Description of the specific application or process the client wants to automate.")
    application_criticality: Optional[str] = Field(None, description="How critical is this application to the client's overall operations? (e.g., low, medium, high). (Corresponds to survey 3.1.1)")
    application_type_and_sub_type: Optional[str] = Field(None, description="Specific type and sub-type of the application. (Corresponds to survey 3.1.2)")
    # Free text inspiration questions from survey section 3 (Part Variety, Size, Weight, Throughput) can be summarized.
    # We can add specific fields if there are common structured answers expected:
    part_variety_description: Optional[str] = Field(None, description="Description of the types and variation of parts/products the automation will handle.")
    part_dimensions_description: Optional[str] = Field(None, description="Typical dimensions of the parts/products involved.")
    part_weight_description: Optional[str] = Field(None, description="Typical weight range of the parts/products.")
    desired_throughput: Optional[str] = Field(None, description="The desired throughput for the automation system (e.g., parts per hour).")


    # Section 4: Commercial Information (Corresponds to survey section 4)
    budget_for_project: Optional[str] = Field(None, description="The client's budget for this automation project. (Corresponds to survey 4.1.1)")
    # Free text inspiration questions from survey section 4 (Budget flexibility, financial constraints)
    project_completion_timeline_ideal: Optional[str] = Field(None, description="Client's ideal timeline for completing this project (e.g., specific date, timeframe). (Corresponds to survey 4.1.2)")
    project_timeline_constraints: Optional[str] = Field(None, description="Specific project timeline constraints or official project timeline category. (Corresponds to survey 4.1.3)")
    key_decision_makers_description: Optional[str] = Field(None, description="Description of key decision-makers involved and their evaluation criteria.")

    # Section 5: General/Other (Placeholder for any other structured info)
    # You can add more fields here based on the "inspiration" questions or if other common structured data emerges.
    # For example, from survey.md:
    company_history_summary: Optional[str] = Field(None, description="Brief summary of the company's history. (Inspired by survey 1.1.1)")
    key_stakeholders_summary: Optional[str] = Field(None, description="Summary of key stakeholders or decision-makers. (Inspired by survey 'Who are the key stakeholders...')")
    company_differentiation_summary: Optional[str] = Field(None, description="Summary of what sets the company apart from competitors. (Inspired by survey 'What sets your company apart...')")
    primary_products_services_summary: Optional[str] = Field(None, description="Summary of primary products or services. (Inspired by survey 'What are the primary products...')")
    innovation_approach_summary: Optional[str] = Field(None, description="Summary of how the company approaches innovation. (Inspired by survey 'How does your company approach innovation...')")
    # This makes it 30+ fields. Adjust as needed.

    class Config:
        # schema_extra can be used to provide an example for documentation or testing
        # schema_extra = {
        #     "example": {
        #         "company_name": "Acme Corp",
        #         "contact_person_name": "John Doe",
        #         "project_description": "Automate widget assembly line",
        #         "budget_for_project": "$50,000 - $75,000",
        #         # ... other example fields
        #     }
        # }
        pass

if __name__ == '__main__':
    # This is for testing and generating the schema if needed
    import json
    schema = LeadReport.model_json_schema()
    print(json.dumps(schema, indent=2))

    # Example usage:
    # report_instance = LeadReport(company_name="Test Inc.", application_description="A new exciting project.")
    # print(report_instance.model_dump_json(indent=2))