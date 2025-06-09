import os
import re
import json
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict

from .llm import call_model_stream, MODELS_BY_PROVIDER

log = logging.getLogger(__name__)

async def generate_rule_updates(
    interaction_summary: str,
    relevant_rules: List[str],
    provider: str,
    model_display_name: str,
    api_key: str = None
) -> List[Dict]:
    """
    Analyzes an interaction and generates a list of proposed rule changes.
    This is the core of the reflective learning loop.
    Returns a list of operation dictionaries, e.g., 
    [{'action': 'update', 'insight': '...', 'old_insight_to_replace': '...'}]
    """
    log.info("Generating rule updates based on interaction...")
    
    insight_sys_prompt = """You are an expert AI knowledge base curator. Your task is to analyze an interaction and output a valid XML structure to update the AI's guiding principles (rules).
The root element must be `<operations_list>`. Each operation is an `<operation>` with child elements: `<action>` (either "add" or "update"), `<insight>` (the new/updated rule text, including its `[TYPE|SCORE]` prefix), and an optional `<old_insight_to_replace>` for "update" actions.
If no changes are needed, output an empty list: `<operations_list></operations_list>`.
**CRITICAL**: Output ONLY the XML structure. No explanations or other text."""
    
    insight_user_prompt = f"Interaction Summary:\n{interaction_summary}\n\nPotentially Relevant Existing Rules:\n{json.dumps(relevant_rules)}\n\nTask: Based on the interaction, generate XML operations to add, update, or consolidate rules to improve the AI's future performance."

    # Allow override for a more powerful "teacher" model
    curator_override = os.getenv("INSIGHT_MODEL_OVERRIDE")
    if curator_override and "/" in curator_override:
        p, m_id = curator_override.split('/', 1)
        models_dict = MODELS_BY_PROVIDER.get(p.lower(), {}).get("models", {})
        m_disp = next((dn for dn, mid in models_dict.items() if mid == m_id), None)
        if m_disp:
            provider, model_display_name = p, m_disp
            api_key = os.getenv(f"{p.upper()}_API_KEY", api_key)
            log.info(f"Using Insight Model Override: {provider}/{model_display_name}")

    messages = [{"role": "system", "content": insight_sys_prompt}, {"role": "user", "content": insight_user_prompt}]
    
    xml_response = ""
    try:
        async for chunk in call_model_stream(provider, model_display_name, messages, api_key, temperature=0.0, max_tokens=2000):
            xml_response += chunk
    except Exception as e:
        log.error(f"Rule generation LLM call failed: {e}")
        return []

    return _parse_rule_update_xml(xml_response)

def _parse_rule_update_xml(xml_string: str) -> List[Dict]:
    """Parses the XML output from the LLM into a list of operation dictionaries."""
    operations = []
    try:
        # Clean the input to find the XML block
        xml_match = re.search(r'<operations_list>.*</operations_list>', xml_string, re.DOTALL | re.IGNORECASE)
        if not xml_match:
            log.warning(f"No <operations_list> found in LLM output. Raw: {xml_string}")
            return []
            
        root = ET.fromstring(xml_match.group(0))
        for op_element in root.findall("operation"):
            action_el = op_element.find("action")
            insight_el = op_element.find("insight")
            old_insight_el = op_element.find("old_insight_to_replace")
            
            action = action_el.text.strip().lower() if action_el is not None and action_el.text else None
            insight = insight_el.text.strip() if insight_el is not None and insight_el.text else None
            old_insight = old_insight_el.text.strip() if old_insight_el is not None and old_insight_el.text else None

            # Validate the structure before appending
            if action and insight and re.match(r"\[(CORE_RULE|RESPONSE_PRINCIPLE|BEHAVIORAL_ADJUSTMENT|GENERAL_LEARNING)\|[\d\.]+\]", insight, re.I):
                operations.append({
                    "action": action,
                    "insight": insight,
                    "old_insight_to_replace": old_insight
                })
            else:
                log.warning(f"Skipped invalid operation from XML. Action: '{action}', Insight: '{insight}'")
    except ET.ParseError as e:
        log.error(f"XML parsing error in learning loop: {e}. XML content that failed:\n{xml_string}")
    
    log.info(f"Parsed {len(operations)} rule operations from LLM.")
    return operations
