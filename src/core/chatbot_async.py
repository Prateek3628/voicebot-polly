"""
Async chatbot orchestrator for parallel processing pipeline.
Coordinates intent, RAG, response generation, and TTS in parallel.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
import time

from core.agent_async import AsyncChatbotAgent, IntentType, get_async_agent
from core.session_manager import SessionManager, session_manager
from core.contact_form_handler import ContactFormHandler
from core.project_context_handler import ProjectContextHandler
from legacy.agent import ContactFormState

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
        self._processing_locks = {}  # Track which sessions are currently processing
        logger.info("AsyncChatBot initialized")
    
    def start_session(self) -> Tuple[str, str]:
        """
        Start a new session.
        Returns: (session_id, welcome_message)
        """
        session_id = self.session_manager.create_session()
        # Set initial state to ask for consent
        self.session_manager.set_contact_form_state(session_id, "asking_initial_consent")
        welcome = "Hello! Welcome to TechGropse. I'm Anup, your virtual assistant. Before we begin, would you like to provide your name and email for a more personalized experience? You can say yes or no."
        logger.info(f"Started session: {session_id} - asking for initial consent")
        return session_id, welcome
    
    def end_session(self, session_id: str):
        """End a session and cleanup."""
        if session_id:
            self.session_manager.clear_session(session_id)
            # Clear conversation history from agent
            self.agent.clear_conversation_history(session_id)
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
            
            # CRITICAL DEDUPLICATION: Check if this exact message was just processed
            try:
                history = self.agent.get_conversation_history(session_id)
                if history and len(history) >= 2:
                    # Check last 2 messages: should be [user message, bot response]
                    last_msg = history[-1]
                    second_last = history[-2] if len(history) >= 2 else None
                    
                    if not last_msg or not second_last:
                        # Skip deduplication if messages are None
                        pass
                    else:
                        # If the second-to-last message is from user and matches current input,
                        # AND last message is from bot (meaning we already processed this)
                        # Then this is a duplicate - return the previous bot response
                        is_bot_last = False
                        if hasattr(last_msg, 'type'):
                            is_bot_last = last_msg.type == 'ai'
                        elif isinstance(last_msg, dict):
                            is_bot_last = last_msg.get('role') == 'bot'
                        
                        if is_bot_last and second_last:
                            user_msg_content = None
                            if hasattr(second_last, 'type') and second_last.type == 'human':
                                user_msg_content = second_last.content if hasattr(second_last, 'content') else None
                            elif isinstance(second_last, dict) and second_last.get('role') == 'user':
                                user_msg_content = second_last.get('content')
                            
                            if user_msg_content and user_msg_content == user_input:
                                logger.warning(f"⚠️ DUPLICATE REQUEST DETECTED: '{user_input[:50]}...' - Returning cached response")
                                # Return the previous bot response
                                cached_response = ''
                                if hasattr(last_msg, 'content'):
                                    cached_response = last_msg.content
                                elif isinstance(last_msg, dict):
                                    cached_response = last_msg.get('content', '')
                                
                                if cached_response:
                                    return {
                                        'response': cached_response,
                                        'intent': 'cached_duplicate',
                                        'timing': {'total': 0.001},
                                        'session_id': session_id,
                                        'duplicate': True
                                    }
            except Exception as e:
                logger.error(f"Error in deduplication check: {e}")
            
            # Append user message to history
            try:
                self.session_manager.append_message_to_history(session_id, 'user', user_input)
            except Exception as e:
                logger.error(f"Error appending message to history: {e}")
            
            # Check contact form state
            form_state = self.session_manager.get_contact_form_state(session_id)
            
            if form_state == ContactFormState.COMPLETED.value:
                self.session_manager.set_contact_form_state(session_id, ContactFormState.IDLE.value)
                form_state = ContactFormState.IDLE.value
            
            # Handle initial consent asking (NEW FLOW)
            if form_state == ContactFormState.ASKING_INITIAL_CONSENT.value:
                user_lower = user_input.lower().strip()
                
                # Check if user is asking about a project instead of answering consent
                is_project_inquiry = any(phrase in user_lower for phrase in [
                    'build an app', 'build a website', 'develop', 'create an app',
                    'make an app', 'need an app', 'cost', 'price', 'timeline'
                ])
                
                if is_project_inquiry:
                    # User is asking about a project, skip consent and help them
                    self.session_manager.set_contact_form_state(session_id, ContactFormState.IDLE.value)
                    # Continue to normal flow below (don't return here)
                # Check for affirmative responses to consent question
                elif any(word in user_lower for word in ['yes', 'yeah', 'yep', 'sure', 'okay', 'ok', 'fine', 'please']):
                    # User wants to provide details - collect name and email in ONE question
                    self.session_manager.set_contact_form_state(session_id, ContactFormState.COLLECTING_NAME_AND_EMAIL.value)
                    return {
                        'response': "Great! Please provide your name and email address.",
                        'intent': 'CONTACT_FORM',
                        'requires_contact_form': True
                    }
                else:
                    # User declined - skip data collection, go straight to helping them
                    self.session_manager.set_contact_form_state(session_id, ContactFormState.IDLE.value)
                    return {
                        'response': "No problem! How can I help you today?",
                        'intent': 'GENERAL',
                        'requires_contact_form': False
                    }
            
            # Handle collecting name and email together (NEW FLOW - handles partial input)
            if form_state == ContactFormState.COLLECTING_NAME_AND_EMAIL.value:
                # Use LLM to extract name and email from user's response
                from langchain_openai import ChatOpenAI
                from config.settings import config
                llm = ChatOpenAI(model=config.openai_model, temperature=0)
                extraction_prompt = f"""Extract the name and email from this user input: "{user_input}"

Return ONLY a JSON object like:
{{"name": "extracted name or null", "email": "extracted email or null"}}

Examples:
- "prateek" → {{"name": "Prateek", "email": null}}
- "My email is prateek@example.com" → {{"name": null, "email": "prateek@example.com"}}
- "John and john@example.com" → {{"name": "John", "email": "john@example.com"}}

If you can't find something, return null for that field."""
                
                try:
                    extraction_result = llm.invoke(extraction_prompt)
                    import json
                    import re
                    
                    result_text = extraction_result.content.strip()
                    # Try to extract JSON from response
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        result_text = json_match.group(0)
                    
                    extracted = json.loads(result_text)
                    
                    name = extracted.get('name')
                    email = extracted.get('email')
                    
                    form_data = self.session_manager.get_contact_form_data(session_id)
                    
                    # Update what we have
                    if name:
                        form_data['name'] = name
                    if email:
                        form_data['email'] = email
                    
                    self.session_manager.set_contact_form_data(session_id, form_data)
                    
                    # Check what's complete
                    has_name = bool(form_data.get('name'))
                    has_email = bool(form_data.get('email'))
                    
                    if has_name and has_email:
                        # Got everything!
                        self.session_manager.set_contact_form_state(session_id, ContactFormState.IDLE.value)
                        return {
                            'response': f"Thank you, {form_data['name']}! I've noted your email as {form_data['email']}. How can I assist you today?",
                            'intent': 'CONTACT_FORM',
                            'requires_contact_form': False
                        }
                    elif has_name and not has_email:
                        # Got name, still need email
                        return {
                            'response': f"Thank you, {form_data['name']}! Could you please also provide your email address?",
                            'intent': 'CONTACT_FORM',
                            'requires_contact_form': True
                        }
                    elif not has_name and has_email:
                        # Got email, still need name
                        return {
                            'response': f"Great! I've noted your email. Could you please also tell me your name?",
                            'intent': 'CONTACT_FORM',
                            'requires_contact_form': True
                        }
                    else:
                        # Didn't extract anything
                        return {
                            'response': "I couldn't quite catch that. Could you please provide your name and email? You can say something like 'My name is John and my email is john@example.com', or provide them one at a time.",
                            'intent': 'CONTACT_FORM',
                            'requires_contact_form': True
                        }
                except Exception as e:
                    logger.error(f"Error extracting name/email: {e}")
                    return {
                        'response': "I couldn't quite catch that. Could you please provide your name first? Just say your name.",
                        'intent': 'CONTACT_FORM',
                        'requires_contact_form': True
                    }
            
            # Handle contact form flow (not parallelized - sequential state machine)
            # Exclude project context states from regular contact form handling
            contact_form_states = [
                ContactFormState.INITIAL_COLLECTING_NAME.value,
                ContactFormState.INITIAL_COLLECTING_EMAIL.value,
                ContactFormState.INITIAL_COLLECTING_PHONE.value,
                ContactFormState.ASKING_CONSENT.value,
                ContactFormState.ASKING_SCHEDULE_CHANGE.value,
                ContactFormState.COLLECTING_DATETIME.value,
                ContactFormState.COLLECTING_TIMEZONE.value,
                ContactFormState.COLLECTING_NAME.value,
                ContactFormState.COLLECTING_EMAIL.value,
                ContactFormState.COLLECTING_PHONE.value
            ]
            
            if form_state in contact_form_states:
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
            
            # Handle project context collection flow (intelligent, not hardcoded)
            if form_state == ContactFormState.COLLECTING_PROJECT_CONTEXT.value:
                project_context = self.session_manager.get_project_context(session_id) or {}
                logger.info(f"Retrieved project context for session {session_id}: {project_context}")
                
                # Get conversation history to avoid asking redundant questions
                conversation_history = []
                try:
                    conversation_history = self.agent.get_conversation_history(session_id)
                except Exception:
                    pass
                
                result = ProjectContextHandler.handle_project_inquiry_intelligently(
                    user_input=user_input,
                    existing_context=project_context,
                    conversation_history=conversation_history
                )
                
                logger.info(f"Project inquiry result: needs_more={result.get('needs_more_context')}, context={result.get('project_context')}")
                
                # Update session based on intelligent analysis
                if result.get('needs_more_context', False):
                    # Still need more context, stay in collecting state
                    self.session_manager.set_contact_form_state(session_id, ContactFormState.COLLECTING_PROJECT_CONTEXT.value)
                else:
                    # Context complete, move to idle
                    self.session_manager.set_contact_form_state(session_id, ContactFormState.IDLE.value)
                
                self.session_manager.set_project_context(session_id, result['project_context'])
                logger.info(f"Saved project context for session {session_id}: {result['project_context']}")
                
                response = result['response']
                
                # CRITICAL: Add to agent's conversation history so it's available for next turn
                self.agent.add_to_conversation_history(session_id, user_input, response)
                
                try:
                    self.session_manager.append_message_to_history(session_id, 'bot', response)
                except Exception:
                    pass
                
                return {
                    'response': response,
                    'intent': 'project_inquiry',
                    'timing': {'total': time.time() - total_start},
                    'session_id': session_id
                }
            
            # CHECK FOR PROJECT INQUIRY FIRST - Before doing expensive RAG!
            # Quick check: Does this look like a project inquiry that needs context collection?
            try:
                user_lower = user_input.lower()
                looks_like_project_inquiry = any(phrase in user_lower for phrase in [
                    'i want to build', 'build an app', 'build a website', 'develop an app', 
                    'create an app', 'make an app', 'need an app', 'looking to build',
                    'cost', 'price', 'timeline', 'how much', 'how long', 'estimate'
                ])
                
                if looks_like_project_inquiry:
                    # Check if we already have context or need to collect it
                    project_context = self.session_manager.get_project_context(session_id) or {}
                    conversation_history = []
                    try:
                        conversation_history = self.agent.get_conversation_history(session_id)
                    except Exception:
                        pass
                    
                    # Analyze context needs
                    result_analysis = ProjectContextHandler.handle_project_inquiry_intelligently(
                        user_input=user_input,
                        existing_context=project_context,
                        conversation_history=conversation_history
                    )
                    
                    if result_analysis.get('needs_more_context', False):
                        # Need to collect more context - DON'T do RAG yet!
                        self.session_manager.set_contact_form_state(
                            session_id, ContactFormState.COLLECTING_PROJECT_CONTEXT.value
                        )
                        response = result_analysis['response']
                        self.session_manager.set_project_context(session_id, result_analysis['project_context'])
                        
                        # Add to agent's conversation history
                        self.agent.add_to_conversation_history(session_id, user_input, response)
                        
                        try:
                            self.session_manager.append_message_to_history(session_id, 'bot', response)
                        except Exception:
                            pass
                        
                        return {
                            'response': response,
                            'intent': 'project_inquiry_collecting',
                            'timing': {'total': time.time() - total_start},
                            'session_id': session_id
                        }
                    else:
                        # We have enough context - NOW do RAG with enhanced query
                        enhanced_query = ProjectContextHandler.get_contextual_prompt(user_input, result_analysis['project_context'])
                        logger.info(f"Context complete, running RAG with enhanced query: {enhanced_query[:100]}...")
                        # Continue to parallel processing below with enhanced query
                        user_input = enhanced_query
            except Exception as e:
                logger.error(f"Error in project inquiry pre-check: {e}", exc_info=True)
                # Continue to normal processing if pre-check fails
            
            # PARALLEL PROCESSING PIPELINE (RAG + Intent + Response)
            result = await self.agent.process_parallel(
                user_input=user_input,
                session_id=session_id,
                fast_mode=fast_mode
            )
            
            response = result.get('response', '')
            intent = result.get('intent', '')
            
            # Legacy handler for project_inquiry intent (if it comes from intent classification)
            # This is a fallback - the above check should catch most cases
            if intent == 'project_inquiry':
                project_context = self.session_manager.get_project_context(session_id) or {}
                
                # Get conversation history to avoid asking redundant questions
                conversation_history = []
                try:
                    conversation_history = self.agent.get_conversation_history(session_id)
                except Exception:
                    pass
                
                # Use intelligent analysis to handle the inquiry
                result_analysis = ProjectContextHandler.handle_project_inquiry_intelligently(
                    user_input=user_input,
                    existing_context=project_context,
                    conversation_history=conversation_history
                )
                
                if result_analysis.get('needs_more_context', False):
                    # Need to collect more context
                    self.session_manager.set_contact_form_state(
                        session_id, ContactFormState.COLLECTING_PROJECT_CONTEXT.value
                    )
                    response = result_analysis['response']
                    self.session_manager.set_project_context(session_id, result_analysis['project_context'])
                    
                    # Add to agent's conversation history
                    self.agent.add_to_conversation_history(session_id, user_input, response)
                else:
                    # We have enough context, enhance the query with context for better RAG results
                    enhanced_query = ProjectContextHandler.get_contextual_prompt(user_input, result_analysis['project_context'])
                    # Re-run RAG with enhanced query
                    enhanced_result = await self.agent.generate_response_with_context(
                        enhanced_query, session_id, result_analysis['project_context']
                    )
                    if enhanced_result:
                        result.update(enhanced_result)
                        response = result.get('response', '')
                    else:
                        response = result_analysis['response']
                    
                    self.session_manager.set_project_context(session_id, result_analysis['project_context'])
                
                # Update result with the intelligent response
                result['response'] = response
            
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
        Process interim (partial) speech with FULL RAG + TTS generation.
        Runs complete pipeline in background while user is still speaking.
        Final response is discarded if user changes their query.
        
        Args:
            partial_text: Partial transcription
            session_id: Session identifier
            
        Returns:
            Dict with full response + pre-generated audio (cached)
        """
        start_time = time.time()
        
        try:
            # Check if partial text is substantial enough (at least 10 chars)
            if len(partial_text) < 10:
                return {
                    'type': 'interim',
                    'intent': 'unknown',
                    'partial_text': partial_text,
                    'session_id': session_id,
                    'ready': False
                }
            
            logger.info(f"🚀 SPECULATIVE EXEC: Processing interim '{partial_text}' while user still speaking")
            
            # Run FULL pipeline (same as final query)
            # This generates complete response + audio in background
            result = await self.process_message_async(
                user_input=partial_text,
                session_id=session_id,
                fast_mode=False  # Full processing
            )
            
            elapsed = time.time() - start_time
            
            logger.info(f"✅ SPECULATIVE EXEC: Complete response ready in {elapsed:.2f}s (cached for final)")
            
            # Mark this as speculative (may be discarded)
            result['type'] = 'speculative'
            result['partial_text'] = partial_text
            result['ready'] = True
            result['speculative_timing'] = elapsed
            
            return result
            
        except Exception as e:
            logger.error(f"Speculative execution error: {e}")
            return {
                'type': 'interim',
                'intent': 'unknown',
                'partial_text': partial_text,
                'session_id': session_id,
                'ready': False,
                'error': str(e)
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
