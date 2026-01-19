"""
Main chatbot interface that orchestrates the conversation flow.
"""
import logging
import sys
from typing import Optional, Dict, Any

from agent import ChatbotAgent, ContactFormState
from session_manager import SessionManager, session_manager
from vectorstore.chromadb_client import ChromaDBClient
from config import config
from contact_form_handler import ContactFormHandler

logger = logging.getLogger(__name__)


class ChatBot:
    """Main chatbot interface that manages the conversation flow."""
    
    def __init__(self):
        """Initialize the chatbot with all required components."""
        try:
            # Initialize components
            self.chromadb_client = ChromaDBClient()
            self.session_manager = session_manager
            self.agent = ChatbotAgent(self.chromadb_client)
            
            # Check if ChromaDB has data
            if self.chromadb_client.is_collection_empty():
                logger.warning("ChromaDB collection is empty. Please run data initialization first.")
                print("\n⚠️  Warning: No data found in ChromaDB. Please run 'python initialise_data.py' first.")
            
            # Note: session_id is NOT stored as instance variable to prevent
            # session mixing in concurrent requests. It must be passed as parameter.
            
            logger.info("ChatBot initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChatBot: {e}")
            raise
    
    def start_session(self) -> tuple[str, str]:
        """
        Start a new chat session and return the session ID and initial message.
        
        IMPORTANT: Session ID is NOT stored as instance variable.
        Caller must store and pass it to other methods.
        
        Returns:
            Tuple of (session_id, initial_message)
        """
        try:
            session_id = self.session_manager.create_session()
            logger.info(f"Started new session: {session_id}")
            
            # Return welcome message asking for name
            initial_message = "Hello! Welcome to TechGropse, I am anup, your virtual assistant. Before we begin, I'd like to know you better. What's your name?"
            return session_id, initial_message
            
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            raise
    
    def end_session(self, session_id: str):
        """
        End the session, save conversation to MongoDB, and clear cache.
        
        Args:
            session_id: Session ID to end
        """
        try:
            if session_id:
                # Check if session exists before ending
                session_info = self.session_manager.get_session_info(session_id)
                if not session_info:
                    logger.info(f"Session {session_id} already ended or doesn't exist")
                    return
                
                # Get conversation history and user details before clearing
                conversation_history = self.session_manager.get_session_history(session_id)
                user_details = self.session_manager.get_contact_form_data(session_id)
                
                # Save to MongoDB if we have mongodb_client
                if self.agent.mongodb_client and conversation_history:
                    try:
                        self.agent.mongodb_client.save_session_conversation(
                            session_id=session_id,
                            conversation_history=conversation_history,
                            user_details=user_details
                        )
                        logger.info(f"Saved conversation for session {session_id} to MongoDB")
                    except Exception as e:
                        logger.error(f"Failed to save conversation to MongoDB: {e}")
                
                # Clear session
                self.session_manager.clear_session(session_id)
                logger.info(f"Ended session: {session_id}")
                print("\n👋 Session ended. Thank you for using TechGropse Virtual Representative!")

            
        except Exception as e:
            logger.error(f"Error ending session: {e}")
    
    def process_message(self, user_input: str, session_id: str) -> str:
        """
        Process a user message through the complete flow.
        
        Args:
            user_input: User's input message
            session_id: Session ID for this request
            
        Returns:
            Bot response
        """
        try:
            if not session_id:
                raise ValueError("session_id is required")

            # Update session activity
            self.session_manager.update_session_activity(session_id)

            # Append user message to session history
            try:
                self.session_manager.append_message_to_history(session_id, 'user', user_input)
            except Exception:
                logger.debug("Unable to append message to history")

            # Check if we're in contact form flow (including initial collection)
            form_state = self.session_manager.get_contact_form_state(session_id)
            
            # If form was just completed, reset to IDLE for next query
            if form_state == ContactFormState.COMPLETED.value:
                self.session_manager.set_contact_form_state(session_id, ContactFormState.IDLE.value)
                form_state = ContactFormState.IDLE.value
                logger.info(f"Contact form completed, resetting to IDLE for session {session_id}")
            
            if form_state != ContactFormState.IDLE.value:
                # Handle contact form step
                form_data = self.session_manager.get_contact_form_data(session_id)
                
                result = ContactFormHandler.handle_contact_form_step(
                    form_state=form_state,
                    user_input=user_input,
                    form_data=form_data,
                    session_id=session_id,
                    mongodb_client=self.agent.mongodb_client
                )
                
                # Update session state
                self.session_manager.set_contact_form_state(session_id, result['next_state'])
                self.session_manager.set_contact_form_data(session_id, result['form_data'])
                
                response = result['response']
                
                # Append bot response to history
                try:
                    self.session_manager.append_message_to_history(session_id, 'bot', response)
                except Exception:
                    logger.debug("Unable to append bot response to history")
                
                logger.info(f"Session {session_id}: ContactForm={form_state}")
                return response

            # Process through agent (provide last user query for follow-ups)
            # Since we already appended current user_input to history, we need to skip it
            # to get the PREVIOUS question (skip_current=True)
            last_user = self.session_manager.get_last_user_query(session_id, skip_current=True)
            result = self.agent.process_user_input(user_input, last_user_query=last_user)

            response = result.get('response', '')
            intent = result.get('intent', '')

            # Check if user explicitly requested contact (new intent type)
            if intent == 'contact_request':
                # User explicitly asked to be contacted
                user_details = self.session_manager.get_contact_form_data(session_id) or {}
                user_details['original_query'] = user_input
                
                # Check if user has existing schedule
                has_schedule = user_details.get('preferred_datetime') and user_details.get('timezone')
                
                if has_schedule:
                    # User has existing schedule, ask if they want to keep or change
                    existing_datetime = user_details.get('preferred_datetime')
                    existing_timezone = user_details.get('timezone')
                    self.session_manager.set_contact_form_data(session_id, user_details)
                    self.session_manager.set_contact_form_state(session_id, ContactFormState.ASKING_SCHEDULE_CHANGE.value)
                    response = f"Sure! You previously scheduled a call for {existing_datetime} ({existing_timezone}). Would you like to keep this time or change it?"
                else:
                    # No existing schedule, ask for availability
                    self.session_manager.set_contact_form_data(session_id, user_details)
                    self.session_manager.set_contact_form_state(session_id, ContactFormState.COLLECTING_DATETIME.value)
                    response = "Great! I'll connect you with our team. When would be the best time for them to reach out to you? Please provide your preferred date and time."
                
                # Append bot response to history
                try:
                    self.session_manager.append_message_to_history(session_id, 'bot', response)
                except Exception:
                    logger.debug("Unable to append bot response to history")
                
                logger.info(f"Session {session_id}: Intent=contact_request")
                return response

            # Check if contact form should be triggered (fallback detection)
            if response == "TRIGGER_CONTACT_FORM":
                # Get existing form data (may have contact info from initial collection)
                existing_form_data = self.session_manager.get_contact_form_data(session_id) or {}
                
                # Add/update original query while preserving existing contact info
                existing_form_data['original_query'] = user_input
                self.session_manager.set_contact_form_data(session_id, existing_form_data)
                
                # Set state to asking consent
                self.session_manager.set_contact_form_state(session_id, ContactFormState.ASKING_CONSENT.value)
                
                # Generate consent message
                response = ContactFormHandler.ask_for_contact_consent(user_input)

            # Append bot response to history
            try:
                self.session_manager.append_message_to_history(session_id, 'bot', response)
            except Exception:
                logger.debug("Unable to append bot response to history")

            # Log interaction
            logger.info(f"Session {session_id}: Intent={intent}")

            return response

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I'm sorry, I encountered an error processing your request. Please try again."
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Args:
            session_id: Session ID to get stats for
            
        Returns:
            Session statistics
        """
        try:
            if not session_id:
                return {"error": "session_id is required"}
            
            session_info = self.session_manager.get_session_info(session_id)
            collection_info = self.chromadb_client.get_collection_info()
            
            return {
                "session_id": session_id,
                "session_info": session_info,
                "collection_info": collection_info
            }
            
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {"error": str(e)}
    
    def run_interactive(self):
        """Run the chatbot in interactive CLI mode."""
        try:
            # Start session and store session_id locally
            session_id = self.start_session()
            
            while True:
                try:
                    user_input = input("\n💬 You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n\n👋 Goodbye! Thanks for using TechGropse Virtual Representative.")
                    break

                if not user_input:
                    # empty input, prompt again
                    continue

                # Normal processing
                try:
                    response = self.process_message(user_input, session_id)
                    
                    # Display response
                    print(f"\n🤖 Bot: {response}")
                    
                    # Check if this was a goodbye intent - if so, end the session and exit
                    # We can check this by looking at the last result intent
                    if user_input.lower() in ["quit", "exit", "bye", "goodbye", "end", "thank you", "thanks", "that's all"]:
                        break
                    
                    # Also check for goodbye patterns in the input
                    goodbye_patterns = ['goodbye', 'bye', 'see you', 'farewell', 'good night', 'take care']
                    if any(pattern in user_input.lower() for pattern in goodbye_patterns):
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing input: {e}")
                    print(f"\n❌ Error: {e}")
                    continue

        finally:
            # Ensure session is cleared on exit
            try:
                self.end_session(session_id)
            except Exception:
                pass
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of all components.
        
        Returns:
            Health status of all components
        """
        health_status = {
            "chatbot": "healthy",
            "session_manager": "unknown",
            "chromadb": "unknown",
            "agent": "unknown"
        }
        
        try:
            # Check session manager (Redis)
            test_session = self.session_manager.create_session()
            if test_session:
                self.session_manager.clear_session(test_session)
                health_status["session_manager"] = "healthy"
            else:
                health_status["session_manager"] = "unhealthy"
        except Exception as e:
            health_status["session_manager"] = f"unhealthy: {e}"
        
        try:
            # Check ChromaDB
            collection_info = self.chromadb_client.get_collection_info()
            if collection_info:
                health_status["chromadb"] = "healthy"
            else:
                health_status["chromadb"] = "unhealthy"
        except Exception as e:
            health_status["chromadb"] = f"unhealthy: {e}"
        
        try:
            # Check agent (basic classification test)
            test_intent = self.agent.classify_intent("hello")
            if test_intent:
                health_status["agent"] = "healthy"
            else:
                health_status["agent"] = "unhealthy"
        except Exception as e:
            health_status["agent"] = f"unhealthy: {e}"
        
        # Overall health
        unhealthy_components = [
            comp for comp, status in health_status.items() 
            if comp != "chatbot" and not status.startswith("healthy")
        ]
        
        if unhealthy_components:
            health_status["chatbot"] = f"partial: {', '.join(unhealthy_components)} unhealthy"
        
        return health_status
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        # Note: Session cleanup is now handled explicitly by callers
        # since session_id is not stored as instance variable
        pass


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        chatbot = ChatBot()
        
        # Check if we should run health check
        if len(sys.argv) > 1 and sys.argv[1] == "--health":
            health = chatbot.health_check()
            print("\n🏥 Health Check Results:")
            for component, status in health.items():
                status_emoji = "✅" if status == "healthy" else "❌"
                print(f"{status_emoji} {component}: {status}")
        else:
            # Run interactive mode
            chatbot.run_interactive()
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
