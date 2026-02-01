import os
import json
from crewai import Agent
from typing import List, Dict, Any

class ClauseSplitterAgent:

    def __init__(self, llm, prompt_manager):
        """
        Initialize the Clause Splitter Agent.
        
        Args:
            llm: Language model instance
            prompt_manager: PromptManager instance for loading provider-specific prompts
        """
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent:
        return Agent(
            role="Legal Document Analyst",
            goal="Break down NDA documents into logical, meaningful clauses",
            backstory="""You are an expert legal analyst with years of experience in contract law 
            and document analysis. You specialize in understanding the structure and components 
            of legal agreements, particularly Non-Disclosure Agreements.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            memory=False
        )

    def split_document(self, document_text: str) -> List[Dict[str, Any]]:
        try:
            prompt = self._load_prompt_template().format(document_text=document_text)
            
            # Create a task for the agent to execute
            from crewai import Task
            task = Task(
                description=prompt,
                expected_output="JSON array of clauses"
            )
            result = self.agent.execute_task(task)

            clauses = self._parse_response(result)

            return clauses
        
        except Exception as e:
            print(f"Error in clause Splitting: {str(e)}")
            return []

    def _load_prompt_template(self) -> str:
        """Load the prompt template using PromptManager."""
        return self.prompt_manager.load_prompt("splitter_prompt")

    def _parse_response(self, response: str):
        try: 
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1

            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                clauses = json.loads(json_str)
                return clauses
            else:
                # Try to find JSON object if array not found
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx]
                    result = json.loads(json_str)
                    # Check if the JSON object contains a clauses array
                    if 'clauses' in result and isinstance(result['clauses'], list):
                        return result['clauses']
                    else:
                        return []
                else:
                    print("No valid JSON found in the response")
                    return []
        
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {str(e)}")
            return []

        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return []

    def validate_clauses(self,clauses:List[Dict[str, Any]]) -> bool:
        required_fields = ['clause_id', 'clause_title', 'clause_text','clause_type']
        for clause in clauses:
            for field in required_fields:
                if field not in clause:
                    print(f"Missing required field '{field}' in clause")
                    return False
                
                if not clause['clause_text'].strip():
                    print(f"Empty clause text for clause {clause.get('clause_id','unknown')}")
                    return False
            
        return True