
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get HuggingFace API token
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM3-3B",
    huggingfacehub_api_token=api_key,
    temperature=0.7,
    max_new_tokens=512,
    task="text-generation"
)

# Wrap for chat interface
chat_model = ChatHuggingFace(llm=llm)

def explain_prediction(prediction, probability, shap_dict):
    """
    Convert SHAP values into human-readable explanation
    
    Args:
        prediction: "Fraudulent" or "Legitimate"
        probability: float (0 to 1), e.g., 0.87 = 87% fraud
        shap_dict: dict of {feature_name: shap_value}
                   Example: {'amt_zscore': 0.35, 'geo_distance': 0.25, ...}
    
    Returns:
        str: Plain English explanation with bullet points
    
    Example:
        >>> shap = {'amt_zscore': 0.35, 'geo_distance': 0.25, 'hour': 0.12}
        >>> explain_prediction("Fraudulent", 0.87, shap)
        "• **Primary Risk**: Transaction amount is highly unusual (2.5 std dev above normal)
         • **Location Risk**: Merchant is 250km from cardholder's usual area
         • **Timing Factor**: Late night transaction (3 AM) adds suspicion
         • **Recommendation**: Block and verify with cardholder immediately"
    """
    
    # Build detailed feature list for the prompt
    feature_text = ""
    for feature, value in shap_dict.items():
        impact = "INCREASES fraud risk" if value > 0 else "DECREASES fraud risk"
        magnitude = "strongly" if abs(value) > 0.3 else "moderately" if abs(value) > 0.1 else "slightly"
        feature_text += f"  • {feature}: {magnitude} {impact} (SHAP value: {value:+.4f})\n"
    
    # Create enhanced prompt for better explainability
    system_message = SystemMessage(content="""
    You are an expert Senior Fraud Analyst at a bank. 
    Your goal is to explain a transaction's risk status to a non-technical branch manager.
    
    GUIDELINES:
    - Focus on the "Story" behind the data (e.g., "high amount in new location").
    - Prioritize the top 2-3 drivers of the decision.
    - Be authoritative but concise (2-3 sentences max).
    - Avoid jargon like "SHAP value" or "z-score"; use natural language (e.g., "unusually high amount").""")
    
    user_message = HumanMessage(content=f"""
TRANSACTION CLASSIFICATION: {prediction}
MODEL CONFIDENCE: {probability:.1%}

CONTRIBUTING FACTORS (ranked by importance):
{feature_text}

Analyze this fraud prediction using the 4-point format:
1. Primary Factor - What's the biggest red flag and why?
2. Secondary Factors - What else is concerning?
3. Risk Assessment - How confident should we be in this prediction?
4. Recommendation - What specific action should the bank take?

Explain each SHAP value in context (e.g., "amt_zscore of +0.35 means the amount is 35% more suspicious than average transactions").

Be detailed but concise. Focus on ACTIONABLE insights.""")
    
    # Get LLM response
    try:
        response = chat_model.invoke([system_message, user_message])
        return response.content.strip()
        
    except Exception as e:
        return f"⚠️ Could not generate explanation: {str(e)}\n\nPlease check your Hugging Face API token."

