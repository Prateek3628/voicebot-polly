"""
Async chatbot orchestrator for parallel processing pipeline.
Coordinates intent, RAG, response generation, and TTS in parallel.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
import time

from agent_async import AsyncChatbotAgent, IntentType, get_async_agent
from session_manager import SessionManager, session_manager
from contact_form_handler import ContactFormHandler
from agent import ContactFormState

logger = logging.getLogger(__name__)


class AsyncChatBot:
    """
    Async chatbot with parallel processing pipeline.
    Target: 2-3 second total response time including TTS.
    """
    
    def __init__(self):
        """Initialize async chatbot with all components."""
        self.agent = get_async_agent()
        self.session_manager = session_manager
        logger.info("AsyncChatBot initialized")
    
    def start_session(self) -> Tuple[str, str]:
        """
        Start a new session.
        Returns: (session_id, welcome_message)
        """
        session_id = self.session_manager.create_session()
        welcome = "Hello! Welcome to TechGropse, I'm Anup, your virtual assistant. What's your name?"
        logger.info(f"Started session: {session_id}")
        return session_id, welcome
    
    def end_session(self, session_id: str):
        """End a session and cleanup."""
        if session_id:
            self.session_manager.clear_session(session_id)
            logger.info(f"Ended session: {session_id}")
    
    async def process_message_async(
        self, 
        user_input: str, 
        session_id: str,
        fast_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Process message with parallel processing pipeline.
        
        Args:
            user_input: User's message
            session_id: Session identifier
            fast_mode: Skip LLM for fastest response
            
        Returns:
            Dict with response, intent, timing info
        """
        total_start = time.time()
        
        try:
            if not session_id:
                raise ValueError("session_id is required")
            
            # Update session activity
            self.session_manager.update_session_activity(session_id)
            
            # Append user message to history
            try:
                self.session_manager.append_message_to_history(session_id, 'user', user_input)
            except Exception:
                pass
            
            # Check contact form state
            form_state = self.session_manager.get_contact_form_state(session_id)
            
            if form_state == ContactFormState.COMPLETED.value:
                self.session_manager.set_contact_form_state(session_id, ContactFormState.IDLE.value)
                form_state = ContactFormState.IDLE.value
            
            # Handle contact form flow (not parallelized - sequential state machine)
            if form_state != ContactFormState.IDLE.value:
                form_data = self.session_manager.get_contact_form_data(session_id)
                result = ContactFormHandler.handle_contact_form_step(
                    form_state=form_state,
                    user_input=user_input,
                    form_data=form_data,
                    session_id=session_id,
                    mongodb_client=None
                )
                
                self.session_manager.set_contact_form_state(session_id, result['next_state'])
                self.session_manager.set_contact_form_data(session_id, result['form_data'])
                
                response = result['response']
                
                try:
                    self.session_manager.append_message_to_history(session_id, 'bot', response)
                except Exception:
                    pass
                
                return {
                    'response': response,
                    'intent': 'contact_form',
                    'timing': {'total': time.time() - total_start},
                    'session_id': session_id
                }
            
            # PARALLEL PROCESSING PIPELINE
            result = await self.agent.process_parallel(
                user_input=user_input,
                fast_mode=fast_mode
            )
            
            response = result.get('response', '')
            intent = result.get('intent', '')
            
            # Check for contact request trigger - immediately ask for availability
            if intent == 'contact_request' or response == "TRIGGER_CONTACT_FORM":
                user_details = self.session_manager.get_contact_form_data(session_id) or {}
                user_details['original_query'] = user_input
                
                has_schedule = user_details.get('preferred_datetime') and user_details.get('timezone')
                
                if has_schedule:
                    self.session_manager.set_contact_form_data(session_id, user_details)
                    self.session_manager.set_contact_form_state(
                        session_id, ContactFormState.ASKING_SCHEDULE_CHANGE.value
                    )
                    response = f"Sure! You previously scheduled a call for {user_details.get('preferred_datetime')} ({user_details.get('timezone')}). Would you like to keep this time or change it?"
                else:
                    self.session_manager.set_contact_form_data(session_id, user_details)
                    self.session_manager.set_contact_form_state(
                        session_id, ContactFormState.COLLECTING_DATETIME.value
                    )
                    response = "Great! I'll connect you with our team. When would be the best time for them to reach out? Please include your timezone and country. For example: 'Tomorrow 3 PM IST India' or 'Monday 10 AM EST USA'"
                
                # Update result with the actual response
                result['response'] = response
            
            # Append bot response to history
            try:
                self.session_manager.append_message_to_history(session_id, 'bot', response)
            except Exception:
                pass
            
            result['session_id'] = session_id
            result['timing']['total'] = time.time() - total_start
            
            logger.info(f"Session {session_id}: Total={result['timing']['total']:.2f}s, Intent={intent}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                'response': "I'm sorry, I encountered an error. Please try again.",
                'intent': 'error',
                'timing': {'total': time.time() - total_start},
                'session_id': session_id
            }
    
    async def process_interim_async(
        self, 
        partial_text: str, 
        session_id: str
    ) -> Dict[str, Any]:
        """
        Process interim (partial) speech for predictive analysis.
        
        Args:
            partial_text: Partial transcription
            session_id: Session identifier
            
        Returns:
            Dict with predicted intent and context preview
        """
        try:
            result = await self.agent.process_interim(partial_text)
            result['session_id'] = session_id
            return result
        except Exception as e:
            logger.error(f"Interim processing error: {e}")
            return {
                'type': 'interim',
                'intent': 'unknown',
                'partial_text': partial_text,
                'session_id': session_id
            }
    
    def process_message_sync(self, user_input: str, session_id: str) -> str:
        """
        Synchronous wrapper for compatibility with existing code.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.process_message_async(user_input, session_id)
            )
            return result.get('response', 'Error processing message')
        finally:
            loop.close()


# Singleton instance
_chatbot_instance = None

def get_async_chatbot() -> AsyncChatBot:
    """Get or create the async chatbot singleton."""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = AsyncChatBot()
    return _chatbot_instance
