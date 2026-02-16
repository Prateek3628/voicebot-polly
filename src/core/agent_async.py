"""
Async CrewAI agent for parallel intent classification, RAG retrieval, and response generation.
Optimized for 2-3 second total response time.
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config import config
from vectorstore.chromadb_client import ChromaDBClient, get_chromadb_client
from utils.reranker import Reranker, get_reranker

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound operations
_executor = ThreadPoolExecutor(max_workers=4)


class IntentType(Enum):
    """Intent classification types."""
    GREETING = "greeting"
    CASUAL_CHAT = "casual_chat"
    FOLLOWUP = "followup"
    CONTACT_REQUEST = "contact_request"
    PROJECT_INQUIRY = "project_inquiry"  # New: for project-related questions
    FEEDBACK = "feedback"
    QUERY = "query"
    GOODBYE = "goodbye"
    UNCLEAR = "unclear"


class AsyncChatbotAgent:
    """
    Async agent with parallel processing for:
    - Intent classification
    - RAG document retrieval
    - Response generation
    - TTS preparation (background)
    """
    
    def __init__(self, chromadb_client: ChromaDBClient = None):
        """Initialize the async chatbot agent."""
        # Use singleton instances to avoid reloading models
        self.chromadb_client = chromadb_client or get_chromadb_client()
        
        # Conversation memory: session_id -> list of LangChain messages
        self.conversation_history = {}
        
        # Initialize reranker if enabled - use singleton
        self.reranker = None
        if config.enable_reranking:
            try:
                self.reranker = get_reranker()
                logger.info(f"Reranker initialized: {config.reranker_model}")
            except Exception as e:
                logger.warning(f"Reranker init failed: {e}")
        
        # Fast LLM for intent classification
        self.fast_llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            openai_api_key=config.openai_api_key,
            max_tokens=50,  # Minimal tokens for intent
            request_timeout=5  # Fast timeout
        )
        
        # Standard LLM for response generation
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            openai_api_key=config.openai_api_key,
            max_tokens=300,
            request_timeout=10
        )
        
        logger.info("AsyncChatbotAgent initialized with parallel processing")
    
    async def classify_intent_async(self, user_input: str) -> IntentType:
        """
        Async intent classification with fast LLM.
        Target: < 500ms
        """
        start = time.time()
        
        try:
            prompt = f"""Classify intent into ONE category:
GREETING - saying hello/hi
CASUAL_CHAT - "I'm doing great", "how are you" (casual)
FOLLOWUP - "tell me more", "elaborate", "need more information", "more details", "can you explain further"
CONTACT_REQUEST - "contact me", "call me", "connect me"
PROJECT_INQUIRY - asking about cost, timeline, budget, "can you build", "how much", development quotes, project estimation, "what would it cost"
FEEDBACK - "tell your team", "share this"
QUERY - asking about services, projects, capabilities, identity
GOODBYE - "bye", "thanks", ending

Input: "{user_input}"

Respond with ONLY the category name:"""

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                _executor,
                lambda: self.fast_llm.invoke(prompt)
            )
            
            intent_text = response.content.strip().upper() if hasattr(response, 'content') else str(response).strip().upper()
            
            elapsed = time.time() - start
            logger.debug(f"Intent classification: {elapsed:.2f}s")
            
            # Map to IntentType
            if 'GREETING' in intent_text:
                return IntentType.GREETING
            elif 'CASUAL' in intent_text:
                return IntentType.CASUAL_CHAT
            elif 'FOLLOWUP' in intent_text:
                return IntentType.FOLLOWUP
            elif 'CONTACT' in intent_text:
                return IntentType.CONTACT_REQUEST
            elif 'PROJECT' in intent_text:
                return IntentType.PROJECT_INQUIRY
            elif 'FEEDBACK' in intent_text:
                return IntentType.FEEDBACK
            elif 'QUERY' in intent_text:
                return IntentType.QUERY
            elif 'GOODBYE' in intent_text:
                return IntentType.GOODBYE
            else:
                return IntentType.UNCLEAR
                
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return self._fallback_intent(user_input)
    
    def _fallback_intent(self, user_input: str) -> IntentType:
        """Fast regex-based fallback for intent classification."""
        user_lower = user_input.lower().strip()
        
        if any(w in user_lower for w in ['hi', 'hello', 'hey']):
            return IntentType.GREETING
        elif any(w in user_lower for w in ['bye', 'goodbye', 'thanks', 'thank you']):
            return IntentType.GOODBYE
        elif any(w in user_lower for w in ['contact me', 'call me', 'connect me']):
            return IntentType.CONTACT_REQUEST
        elif any(w in user_lower for w in ['cost', 'price', 'budget', 'how much', 'can you build', 'timeline', 'estimate']):
            return IntentType.PROJECT_INQUIRY
        elif any(w in user_lower for w in ['more info', 'tell me more', 'elaborate']):
            return IntentType.FOLLOWUP
        return IntentType.QUERY
    
    async def retrieve_documents_async(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Async document retrieval from ChromaDB with query enhancement for better tech stack results.
        Target: < 300ms
        """
        start = time.time()
        
        try:
            # Enhance query intelligently using LLM for better retrieval
            enhanced_query = await self._enhance_query_intelligently(query)
            
            loop = asyncio.get_event_loop()
            
            # Run ChromaDB search in executor
            initial_n = config.rerank_candidates if self.reranker else n_results
            results = await loop.run_in_executor(
                _executor,
                lambda: self.chromadb_client.search_similar_documents(enhanced_query, initial_n)
            )
            
            # Filter by distance threshold
            filtered = [r for r in results if r.get('distance', 0) < 1.7]
            
            # Rerank if enabled
            if self.reranker and filtered:
                filtered = await loop.run_in_executor(
                    _executor,
                    lambda: self.reranker.rerank(query, filtered, config.rerank_top_k)
                )
            
            elapsed = time.time() - start
            logger.debug(f"Document retrieval: {elapsed:.2f}s, {len(filtered)} docs")
            
            return filtered[:n_results]
            
        except Exception as e:
            logger.error(f"Document retrieval error: {e}")
            return []
    
    async def _enhance_query_intelligently(self, query: str) -> str:
        """
        Intelligently enhance queries using LLM with manual fallback.
        This replaces hardcoded keyword matching with intelligent understanding.
        """
        query_lower = query.lower()
        
        # Manual fallback for tech stack queries (when LLM fails)
        tech_indicators = [
            'tech stack', 'technology stack', 'technologies', 'programming languages',
            'frameworks', 'tools', 'development stack', 'technical capabilities',
            'what do you use', 'what does techgropse use', 'programming'
        ]
        
        # Exclude pricing/cost queries from tech enhancement
        pricing_indicators = ['cost', 'price', 'pricing', 'budget', 'how much']
        has_pricing = any(indicator in query_lower for indicator in pricing_indicators)
        
        if any(indicator in query_lower for indicator in tech_indicators) and not has_pricing:
            # Direct enhancement for tech stack queries
            enhancement = " React Native Flutter Node.js Python MongoDB Kotlin Swift technology stack programming languages frameworks"
            enhanced_query = f"{query}{enhancement}"
            logger.debug(f"Enhanced tech query: '{query}' -> '{enhanced_query}'")
            return enhanced_query
        
        # Try LLM enhancement for other queries
        try:
            prompt = f"""Enhance this query with relevant search terms:

Query: "{query}"

If about pricing/cost: add " pricing cost development budget"
If about team/company: add " team developers company"
If about services: add " mobile app development services"

Enhanced:"""

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                _executor,
                lambda: self.fast_llm.invoke(prompt)
            )
            
            enhanced = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            # Safety check - if enhancement seems wrong, return original
            if len(enhanced) > len(query) * 3:  # Too long, probably an error
                return query
                
            return enhanced if enhanced else query
            
        except Exception as e:
            logger.debug(f"Query enhancement failed: {e}")
            return query
    
    async def generate_response_async(
        self, 
        query: str, 
        intent: IntentType, 
        context_docs: List[Dict[str, Any]],
        session_id: str = None,
        fast_mode: bool = False
    ) -> str:
        """
        Async response generation.
        fast_mode: Skip LLM, return top doc directly (< 100ms)
        normal_mode: Use LLM for contextual response (< 1s)
        """
        start = time.time()
        
        try:
            # Handle non-query intents quickly
            if intent == IntentType.GREETING:
                return await self._handle_greeting_async(query)
            elif intent == IntentType.CASUAL_CHAT:
                return await self._handle_casual_async(query)
            elif intent == IntentType.GOODBYE:
                return "Thanks for chatting! Feel free to reach out anytime - we at TechGropse are always here to help!"
            elif intent == IntentType.FEEDBACK:
                return "Got it! I'll make sure to pass this along to our team at TechGropse."
            elif intent == IntentType.UNCLEAR:
                return "I'm not quite sure I follow. Could you tell me a bit more about what you're looking for?"
            elif intent == IntentType.FOLLOWUP:
                # Handle followup questions with conversation context
                return await self._handle_followup_async(query, context_docs, session_id)
            elif intent == IntentType.CONTACT_REQUEST:
                # Return trigger signal - chatbot_async will handle asking for availability
                return "TRIGGER_CONTACT_FORM"
            
            # FAST MODE: Return top document directly
            if fast_mode and context_docs:
                content = context_docs[0].get('content', '')
                # Return first 3 sentences
                sentences = content.split('.')[:3]
                elapsed = time.time() - start
                logger.debug(f"Fast mode response: {elapsed:.2f}s")
                return '. '.join(sentences) + '.'
            
            # NORMAL MODE: LLM-powered contextual response
            if not context_docs:
                return "I don't have specific information about that in our documents. Would you like me to connect you with our team?"
            
            # Build context
            context_text = "\n\n".join([
                f"[{doc.get('metadata', {}).get('source', 'Unknown')}]\n{doc['content'][:400]}"
                for doc in context_docs[:3]
            ])
            
            # Create combined prompt with context and use LangChain message format
            combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{context_text}

CRITICAL INSTRUCTIONS:
- You MUST answer using ONLY the information provided in the documents above
- If the question is about technology, tech stack, or tools - list the specific technologies mentioned in the documents
- Answer in 2-3 sentences max but include specific technical details from the documents
- Be conversational and warm
- Use "we at TechGropse", "our" when referring to company
- You can help with services, projects, AI solutions, app development, and scheduling meetings
- Do NOT give generic responses - use the specific information from the documents
- No greetings like "Hi!"

If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""
            
            # Get conversation history and create messages
            chat_history = self.get_conversation_history(session_id)
            
            messages = [
                SystemMessage(content="You are Anup, TechGropse's friendly virtual assistant. Answer questions based on provided documents and conversation history."),
            ] + chat_history + [
                HumanMessage(content=combined_input)
            ]
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                _executor,
                lambda: self.llm.invoke(messages)
            )
            
            result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            elapsed = time.time() - start
            logger.debug(f"LLM response: {elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I'm having trouble processing that right now. Please try again."
    
    async def _handle_greeting_async(self, user_input: str) -> str:
        """Quick greeting response."""
        if any(p in user_input.lower() for p in ['how are you', 'how r u']):
            return "I'm doing great, thanks for asking! How about you?"
        return "Hi! I'm Anup from TechGropse. How can I help you today?"
    
    async def _handle_casual_async(self, user_input: str) -> str:
        """Quick casual chat response."""
        if any(p in user_input.lower() for p in ['how are you', 'how r u']):
            return "I'm doing great, thanks! How about you?"
        return "That's wonderful to hear! What can I help you with today?"
    
    async def _handle_followup_async(self, query: str, context_docs: List[Dict[str, Any]], session_id: str) -> str:
        """Handle followup questions using conversation history in LangChain format."""
        chat_history = self.get_conversation_history(session_id)
        
        if not chat_history:
            # No conversation context, treat as regular query
            if not context_docs:
                return "I don't have specific information about that. Would you like me to connect you with our team?"
            
            # Use first document as context
            context_text = context_docs[0].get('content', '')[:400]
            return f"Based on what I know: {context_text.split('.')[0]}. Would you like more details about this?"
        
        # Build context from documents
        context_text = "\n\n".join([
            f"[{doc.get('metadata', {}).get('source', 'Unknown')}]\n{doc['content'][:400]}"
            for doc in context_docs[:3]
        ])
        
        # Create combined input for followup handling
        combined_input = f"""The user is asking a follow-up question: {query}

Based on our conversation history and these documents, please provide more detailed information:

Documents:
{context_text}

Instructions:
- This is a follow-up question about the previous topic in our conversation
- Look at the conversation history to understand what they previously asked about
- Provide more detailed, specific information about that topic
- Use the documents to give accurate details
- Be conversational and warm  
- Use "we at TechGropse", "our" when referring to company
- 2-3 sentences max

If you can't provide more details based on the documents, say "I don't have additional details about that in our documents."
"""
        
        try:
            # Use conversation history with the followup prompt
            messages = [
                SystemMessage(content="You are Anup, TechGropse's friendly virtual assistant. Handle follow-up questions by looking at conversation history and provided documents."),
            ] + chat_history + [
                HumanMessage(content=combined_input)
            ]
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                _executor,
                lambda: self.llm.invoke(messages)
            )
            
            result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            return result
            
        except Exception as e:
            logger.error(f"Followup response error: {e}")
            return "I'd be happy to provide more details. Could you be more specific about what aspect you'd like to know more about?"
    
    async def _rewrite_query_with_history(self, user_input: str, session_id: str) -> str:
        """
        Rewrite user query to be standalone using conversation history.
        Based on the reference implementation for better context awareness.
        """
        if not session_id or session_id not in self.conversation_history:
            return user_input
            
        chat_history = self.conversation_history[session_id]
        if not chat_history:
            return user_input
            
        try:
            # Create messages for query rewriting
            messages = [
                SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Include relevant context from the conversation. Just return the rewritten question."),
            ] + chat_history[-6:] + [  # Use last 6 messages for context
                HumanMessage(content=f"New question: {user_input}")
            ]
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _executor,
                lambda: self.fast_llm.invoke(messages)
            )
            
            rewritten_query = result.content.strip() if hasattr(result, 'content') else str(result).strip()
            logger.debug(f"Query rewritten: '{user_input}' -> '{rewritten_query}'")
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            return user_input
    
    def add_to_conversation_history(self, session_id: str, user_input: str, response: str):
        """Add messages to conversation history using LangChain format."""
        if not session_id:
            return
            
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        # Add both user and AI messages
        self.conversation_history[session_id].append(HumanMessage(content=user_input))
        self.conversation_history[session_id].append(AIMessage(content=response))
        
        # Keep only the last 12 messages (6 pairs)
        if len(self.conversation_history[session_id]) > 12:
            self.conversation_history[session_id] = self.conversation_history[session_id][-12:]
    
    def get_conversation_history(self, session_id: str) -> List:
        """Get LangChain message history for a session. Returns last 12 messages (6 pairs)."""
        if not session_id or session_id not in self.conversation_history:
            return []
        return self.conversation_history[session_id][-12:]  # Return last 12 messages = 6 user-bot pairs
    
    def clear_conversation_history(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
    
    async def process_parallel(
        self, 
        user_input: str,
        session_id: str = None,
        fast_mode: bool = False,
        skip_intent: bool = False,
        predicted_intent: IntentType = None
    ) -> Dict[str, Any]:
        """
        PARALLEL PROCESSING PIPELINE
        
        Runs Intent Classification and RAG Retrieval simultaneously.
        Target total time: 1-2 seconds
        
        Args:
            user_input: User's message
            fast_mode: Skip LLM for instant response
            skip_intent: Use predicted_intent instead of classifying
            predicted_intent: Pre-predicted intent (for interim processing)
        
        Returns:
            Dict with intent, response, context_docs, timing
        """
        total_start = time.time()
        timing = {}
        
        try:
            # STEP 0: Query rewriting with conversation history
            if session_id and not skip_intent:  # Don't rewrite for interim processing
                rewrite_start = time.time()
                search_query = await self._rewrite_query_with_history(user_input, session_id)
                timing['query_rewrite'] = time.time() - rewrite_start
            else:
                search_query = user_input
            
            # STEP 1: PARALLEL - Intent + RAG simultaneously
            parallel_start = time.time()
            
            if skip_intent and predicted_intent:
                # Use pre-predicted intent
                intent = predicted_intent
                # Only run RAG with the search query
                context_docs = await self.retrieve_documents_async(search_query)
                timing['parallel'] = time.time() - parallel_start
            else:
                # Run both in parallel - use original user_input for intent, search_query for RAG
                intent_task = asyncio.create_task(self.classify_intent_async(user_input))
                rag_task = asyncio.create_task(self.retrieve_documents_async(search_query))
                
                intent, context_docs = await asyncio.gather(intent_task, rag_task)
                timing['parallel'] = time.time() - parallel_start
            
            logger.info(f"Parallel phase: {timing['parallel']:.2f}s (Intent: {intent.value}, Docs: {len(context_docs)})")
            
            # STEP 2: Response Generation
            response_start = time.time()
            response = await self.generate_response_async(
                user_input, 
                intent, 
                context_docs,
                session_id=session_id,
                fast_mode=fast_mode
            )
            timing['response'] = time.time() - response_start
            
            timing['total'] = time.time() - total_start
            
            logger.info(f"Total processing: {timing['total']:.2f}s")
            
            # Store in conversation history
            if session_id and response:
                self.add_to_conversation_history(session_id, user_input, response)
            
            return {
                'intent': intent.value,
                'response': response,
                'context_docs': context_docs,
                'timing': timing,
                'user_input': user_input
            }
            
        except Exception as e:
            logger.error(f"Parallel processing error: {e}")
            return {
                'intent': 'error',
                'response': "I encountered an error. Please try again.",
                'context_docs': [],
                'timing': {'total': time.time() - total_start},
                'user_input': user_input
            }
    
    async def process_interim(self, partial_text: str) -> Dict[str, Any]:
        """
        Process interim (partial) speech for predictive analysis.
        Returns quick predictions without full response generation.
        
        Target: < 500ms
        """
        start = time.time()
        
        try:
            # Run intent classification only for interim
            intent = await self.classify_intent_async(partial_text)
            
            # For QUERY intent, start RAG prefetch in background
            context_preview = []
            if intent == IntentType.QUERY:
                # Quick RAG with minimal results
                context_preview = await self.retrieve_documents_async(partial_text, n_results=2)
            
            return {
                'type': 'interim',
                'intent': intent.value,
                'partial_text': partial_text,
                'context_preview': [
                    {'section': doc.get('metadata', {}).get('source', 'Unknown')[:50]}
                    for doc in context_preview
                ],
                'timing': time.time() - start
            }
            
        except Exception as e:
            logger.error(f"Interim processing error: {e}")
            return {
                'type': 'interim',
                'intent': 'unknown',
                'partial_text': partial_text,
                'timing': time.time() - start
            }
    
    async def generate_response_with_context(
        self, 
        enhanced_query: str, 
        session_id: str, 
        project_context: dict
    ) -> dict:
        """
        Generate response using project context for enhanced RAG results.
        
        Args:
            enhanced_query: Query enhanced with project context
            session_id: Session identifier
            project_context: Collected project context
            
        Returns:
            Enhanced response dictionary
        """
        try:
            start_time = time.time()
            
            # Use parallel processing for enhanced query
            result = await self.process_parallel(
                user_input=enhanced_query,
                session_id=session_id,
                fast_mode=False
            )
            
            # Add project context info to response metadata
            result['project_context_used'] = True
            result['project_type'] = project_context.get('project_type')
            result['project_sector'] = project_context.get('project_sector')
            
            logger.info(f"Enhanced response with project context: {time.time() - start_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Enhanced response generation error: {e}")
            return None


# Singleton instance for reuse
_agent_instance = None

def get_async_agent() -> AsyncChatbotAgent:
    """Get or create the async agent singleton."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AsyncChatbotAgent()
    return _agent_instance
