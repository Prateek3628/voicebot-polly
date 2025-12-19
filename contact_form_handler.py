"""
Contact form handling methods for ChatbotAgent.
These methods handle the multi-step contact request form collection.
"""
import logging
from typing import Dict, Any, Optional
from utils.validators import validate_email, validate_phone, validate_datetime, validate_name
from database.mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)


class ContactFormHandler:
    """Handles contact form state and collection logic."""
    
    @staticmethod
    def should_trigger_contact_form(context_docs: list, distance_threshold: float = 1.5) -> bool:
        """
        Determine if contact form should be triggered based on search results.
        
        Args:
            context_docs: Retrieved context documents
            distance_threshold: Maximum acceptable distance for relevant results
            
        Returns:
            True if contact form should be triggered
        """
        # Trigger if no results or all results have poor similarity
        if not context_docs:
            return True
        
        # Check if all results are beyond the threshold
        relevant_count = sum(1 for doc in context_docs if doc.get('distance', 0) < distance_threshold)
        return relevant_count == 0
    
    @staticmethod
    def ask_for_contact_consent(original_query: str, is_explicit_request: bool = False) -> str:
        """
        Generate message asking user if they want to be contacted.
        
        Args:
            original_query: The query that triggered this
            is_explicit_request: True if user explicitly asked to be contacted (e.g., "connect me")
            
        Returns:
            Consent request message
        """
        if is_explicit_request:
            # User explicitly asked to be contacted - skip the "I don't have info" part
            return """Sure! I'd be happy to connect you with our team. Let me collect a few details so they can reach out to you.

What's your name?"""
        else:
            # RAG fallback - no information found
            return f"""I don't have specific information about that in our current documents. However, I'd be happy to connect you with our team who can provide detailed assistance with your question about: "{original_query}"

Would you like us to contact you? Just reply with 'yes' or 'no'."""
    
    @staticmethod
    def handle_contact_form_step(
        form_state: str,
        user_input: str,
        form_data: Dict[str, Any],
        session_id: str,
        mongodb_client: Optional[MongoDBClient]
    ) -> Dict[str, Any]:
        """
        Handle a single step of the contact form collection.
        
        Args:
            form_state: Current form state
            user_input: User's input
            form_data: Partially collected form data
            session_id: Session ID
            mongodb_client: MongoDB client instance
            
        Returns:
            Dictionary with next_state, response, and updated form_data
        """
        from agent import ContactFormState
        
        user_input = user_input.strip()
        
        # Handle INITIAL collection (at session start)
        if form_state == ContactFormState.INITIAL_COLLECTING_NAME.value:
            is_valid, error = validate_name(user_input)
            if not is_valid:
                return {
                    'next_state': form_state,
                    'response': f"{error} Please provide your full name:",
                    'form_data': form_data
                }
            form_data['name'] = user_input
            return {
                'next_state': ContactFormState.INITIAL_COLLECTING_EMAIL.value,
                'response': f"Thanks, {user_input}! What's your email address?",
                'form_data': form_data
            }
        
        elif form_state == ContactFormState.INITIAL_COLLECTING_EMAIL.value:
            is_valid, error = validate_email(user_input)
            if not is_valid:
                return {
                    'next_state': form_state,
                    'response': f"{error} Please provide a valid email address:",
                    'form_data': form_data
                }
            form_data['email'] = user_input
            return {
                'next_state': ContactFormState.INITIAL_COLLECTING_PHONE.value,
                'response': "Perfect! What's your mobile number? Please include your country code.",
                'form_data': form_data
            }
        
        elif form_state == ContactFormState.INITIAL_COLLECTING_PHONE.value:
            is_valid, error = validate_phone(user_input)
            if not is_valid:
                return {
                    'next_state': form_state,
                    'response': f"{error} Please provide your mobile number:",
                    'form_data': form_data
                }
            form_data['mobile'] = user_input
            return {
                'next_state': ContactFormState.IDLE.value,
                'response': f"Thank you! I now have your details. How can I assist you today?",
                'form_data': form_data  # Keep the data for later use
            }
        
        # Handle consent
        if form_state == ContactFormState.ASKING_CONSENT.value:
            if user_input.lower() in ['yes', 'y', 'sure', 'ok', 'okay', 'yeah']:
                return {
                    'next_state': ContactFormState.COLLECTING_NAME.value,
                    'response': "Great! Let me collect a few details. What's your full name?",
                    'form_data': form_data
                }
            else:
                return {
                    'next_state': ContactFormState.IDLE.value,
                    'response': "No problem! Is there anything else I can help you with?",
                    'form_data': {}
                }
        
        # Collect name
        elif form_state == ContactFormState.COLLECTING_NAME.value:
            is_valid, error = validate_name(user_input)
            if not is_valid:
                return {
                    'next_state': form_state,
                    'response': f"{error} Please provide your full name:",
                    'form_data': form_data
                }
            form_data['name'] = user_input
            return {
                'next_state': ContactFormState.COLLECTING_EMAIL.value,
                'response': f"Thanks, {user_input}! What's your email address?",
                'form_data': form_data
            }
        
        # Collect email
        elif form_state == ContactFormState.COLLECTING_EMAIL.value:
            is_valid, error = validate_email(user_input)
            if not is_valid:
                return {
                    'next_state': form_state,
                    'response': f"{error} Please provide a valid email address:",
                    'form_data': form_data
                }
            form_data['email'] = user_input
            return {
                'next_state': ContactFormState.COLLECTING_PHONE.value,
                'response': "Perfect! What's your mobile number? Please include your country code.",
                'form_data': form_data
            }
        
        # Collect phone
        elif form_state == ContactFormState.COLLECTING_PHONE.value:
            is_valid, error = validate_phone(user_input)
            if not is_valid:
                return {
                    'next_state': form_state,
                    'response': f"{error} Please provide your mobile number:",
                    'form_data': form_data
                }
            form_data['mobile'] = user_input
            return {
                'next_state': ContactFormState.COLLECTING_DATETIME.value,
                'response': "Got it! When would you prefer us to contact you? You can specify in any format.",
                'form_data': form_data
            }
        
        # Collect datetime
        elif form_state == ContactFormState.COLLECTING_DATETIME.value:
            is_valid, error = validate_datetime(user_input)
            if not is_valid:
                return {
                    'next_state': form_state,
                    'response': f"{error} Please provide your preferred date and time:",
                    'form_data': form_data
                }
            form_data['preferred_datetime'] = user_input
            return {
                'next_state': ContactFormState.COLLECTING_TIMEZONE.value,
                'response': "Great! What's your timezone? (e.g., IST, UTC+5:30, EST, PST, GMT)",
                'form_data': form_data
            }
        
        # Collect timezone and complete
        elif form_state == ContactFormState.COLLECTING_TIMEZONE.value:
            from utils.validators import validate_timezone
            
            is_valid, error = validate_timezone(user_input)
            if not is_valid:
                return {
                    'next_state': form_state,
                    'response': f"{error} Please provide your timezone:",
                    'form_data': form_data
                }
            form_data['timezone'] = user_input
            
            # Save to MongoDB
            if mongodb_client:
                try:
                    request_id = mongodb_client.create_contact_request(
                        session_id=session_id,
                        name=form_data['name'],
                        email=form_data['email'],
                        mobile=form_data['mobile'],
                        preferred_datetime=form_data['preferred_datetime'],
                        timezone=form_data['timezone'],
                        original_query=form_data.get('original_query', 'Not specified')
                    )
                    if request_id:
                        logger.info(f"Contact request saved: {request_id}")
                    else:
                        logger.error("Failed to save contact request")
                except Exception as e:
                    logger.error(f"Error saving contact request: {e}")
            
            return {
                'next_state': ContactFormState.COMPLETED.value,
                'response': "All set! We've recorded your request and our team will contact you. Is there anything else I can help you with?",
                'form_data': {}  # Clear form data after completion
            }
        
        # Default fallback
        return {
            'next_state': ContactFormState.IDLE.value,
            'response': "Something went wrong. Let's start over. How can I help you?",
            'form_data': {}
        }
