"""
Project context handling methods for ChatbotAgent.
These methods handle the multi-step project context collection for better responses.
"""
import logging
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from config.settings import config

logger = logging.getLogger(__name__)


class ProjectContextHandler:
    """Handles project context collection for personalized responses."""
    
    # LLM instance for understanding natural language responses
    _llm = None
    
    @classmethod
    def get_llm(cls):
        """Get or create LLM instance."""
        if cls._llm is None:
            cls._llm = ChatOpenAI(
                model=config.openai_model,
                temperature=0.1,
                api_key=config.openai_api_key
            )
        return cls._llm
    
    @staticmethod
    def analyze_project_context_completeness(user_input: str, existing_context: dict, conversation_history: list = None) -> Dict[str, Any]:
        """
        Intelligently analyze if we have enough project context and what questions to ask.
        Uses conversation history to avoid asking redundant questions.
        
        Args:
            user_input: User's current input
            existing_context: Current project context
            conversation_history: List of recent conversation messages
            
        Returns:
            Dictionary with analysis results, next question (if needed), and updated context
        """
        try:
            # Get or create LLM instance with optimized settings for speed
            llm = ProjectContextHandler.get_llm()
            
            # Quick optimization: If user input is very short, do fast simple extraction
            if len(user_input) < 50 and not existing_context:
                # User said something short like "I want to build an app"
                # Skip complex analysis, just extract basics quickly
                simple_result = {
                    "has_enough_context": False,
                    "extracted_info": {},
                    "missing_info": ["project details"],
                    "next_question": "That sounds interesting! Could you tell me more about your project? What type of application are you thinking of - mobile app, website, or both? And what industry is it for?",
                    "reasoning": "Quick response for simple query"
                }
                return simple_result
            
            # Prepare context summary
            context_summary = ""
            if existing_context:
                context_parts = []
                if existing_context.get('project_type'):
                    context_parts.append(f"Type: {existing_context['project_type']}")
                if existing_context.get('project_sector'):
                    context_parts.append(f"Industry: {existing_context['project_sector']}")
                if existing_context.get('project_features'):
                    features = ', '.join(existing_context['project_features'][:5])
                    context_parts.append(f"Features: {features}")
                context_summary = " | ".join(context_parts)
            
            # Prepare conversation history summary (limit to last 5 for performance)
            history_summary = ""
            if conversation_history:
                recent_messages = conversation_history[-5:]  # Reduced from 10 to 5 for speed
                history_parts = []
                for msg in recent_messages:
                    if isinstance(msg, dict):
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')[:150]  # Reduced from 200 to 150
                        history_parts.append(f"{role}: {content}")
                history_summary = " | ".join(history_parts)

            prompt = f"""You are analyzing a user's project inquiry to determine what follow-up questions are needed.
CRITICAL: Use conversation history to avoid asking questions that have already been answered or discussed.

Current user input: "{user_input}"
Existing context: {context_summary or "None"}
Recent conversation: {history_summary or "No previous conversation"}

IMMEDIATE RECOGNITION RULES:
🚨 IF user input contains DETAILED SPECIFICATIONS like numbered requirements, feature lists, or comprehensive project descriptions:
   → ALWAYS set has_enough_context: true
   → Extract ALL mentioned features
   → Set next_question: null

🚨 IF user already answered "both" or "website and mobile" in conversation history:
   → Don't ask about project type again
   → Focus on what they need help with

🚨 IF user provided extensive requirements (like chess platform, tournament management, user roles, payment systems):
   → Recognize this as a COMPLETE project specification
   → Set has_enough_context: true immediately

CONTEXT SUFFICIENCY RULES:
✅ SUFFICIENT CONTEXT = has_enough_context: true when:
- User provided detailed project specifications with multiple features
- User confirmed project type (website, mobile, both) + industry + features
- User says "no additional features" after providing requirements
- ANY comprehensive requirements document is provided

⚠️ INSUFFICIENT CONTEXT = has_enough_context: false ONLY when:
- User said something vague like "I want an app" with NO details
- Genuinely missing critical information

SPECIAL RECOGNITION PATTERNS:
- "Chess Tournament & League Management Platform" + detailed features = COMPLETE specification
- "both website and mobile" + detailed requirements = COMPLETE specification  
- Multiple numbered requirements (1., 2., 3., etc.) = COMPLETE specification
- User management + payment systems + specific features = COMPLETE specification

RESPONSE FORMAT (JSON):
{{
    "has_enough_context": true/false,
    "extracted_info": {{
        "project_type": "extracted type or null",
        "project_sector": "extracted sector or null", 
        "project_features": ["comprehensive list of features extracted"]
    }},
    "missing_info": ["list of missing key information"],
    "next_question": "specific question to ask or null if enough context",
    "reasoning": "detailed explanation"
}}

CRITICAL: If user provided ANY detailed requirements or specifications, ALWAYS set has_enough_context: true

Respond with ONLY valid JSON:"""

            response = llm.invoke(prompt)
            result_text = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            # Parse JSON response - try to extract JSON even if there's extra text
            import json
            try:
                # First try direct parsing
                result = json.loads(result_text)
                return result
            except json.JSONDecodeError:
                # Try to find JSON within the response (sometimes LLM adds extra text)
                import re
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        result = json.loads(json_str)
                        return result
                    except json.JSONDecodeError:
                        pass
                
                # Intelligent fallback - use LLM to generate appropriate response
                logger.error(f"Failed to parse LLM JSON response: {result_text}")
                
                # Try to extract at least basic info using simple LLM call
                try:
                    fallback_llm = ProjectContextHandler.get_llm()
                    fallback_prompt = f"""The user said: "{user_input}"
                    
Based on their input, provide a helpful response. If they provided detailed information, acknowledge it and ask what they'd like to know. If they were vague, ask for specific clarification.

Respond naturally as a helpful assistant:"""
                    
                    fallback_response = fallback_llm.invoke(fallback_prompt)
                    intelligent_response = fallback_response.content.strip() if hasattr(fallback_response, 'content') else str(fallback_response).strip()
                    
                    return {
                        "has_enough_context": False,
                        "extracted_info": {},
                        "missing_info": ["clarification needed"],
                        "next_question": intelligent_response,
                        "reasoning": "JSON parsing failed, used intelligent fallback"
                    }
                except Exception:
                    # Last resort - but still try to be contextual
                    contextual_response = f"I want to help you with your project. Could you tell me more about what you're looking to build?"
                    return {
                        "has_enough_context": False,
                        "extracted_info": {},
                        "missing_info": ["project details"],
                        "next_question": contextual_response,
                        "reasoning": "All parsing failed, using minimal fallback"
                    }
                
        except Exception as e:
            logger.error(f"Context analysis error: {e}")
            
            # Intelligent error fallback - try to respond based on user input
            try:
                fallback_llm = ProjectContextHandler.get_llm()
                error_prompt = f"""There was a technical error, but I still want to help the user who said: "{user_input}"

Provide a helpful response that acknowledges their input and asks for clarification in a natural way:"""
                
                error_response = fallback_llm.invoke(error_prompt)
                intelligent_error_response = error_response.content.strip() if hasattr(error_response, 'content') else str(error_response).strip()
                
                return {
                    "has_enough_context": False,
                    "extracted_info": {},
                    "missing_info": ["clarification needed"],
                    "next_question": intelligent_error_response,
                    "reasoning": f"Error in analysis: {e}, used intelligent fallback"
                }
            except Exception:
                # Absolute last resort
                return {
                    "has_enough_context": False,
                    "extracted_info": {},
                    "missing_info": ["project details"],
                    "next_question": "I'd love to help with your project! What are you looking to build?",
                    "reasoning": f"Critical error: {e}"
                }
    
    @staticmethod
    def handle_project_inquiry_intelligently(
        user_input: str,
        existing_context: dict,
        conversation_history: list = None
    ) -> Dict[str, Any]:
        """
        Intelligently handle project inquiry without hardcoded states.
        Uses conversation history to avoid redundant questions.
        
        Args:
            user_input: User's input
            existing_context: Current project context
            conversation_history: Recent conversation messages
            
        Returns:
            Dictionary with response, updated context, and completion status
        """
        try:
            # Check if existing context is already sufficient
            has_type = bool(existing_context.get('project_type'))
            has_sector = bool(existing_context.get('project_sector'))
            has_features = len(existing_context.get('project_features', [])) >= 3
            
            # If we already have sufficient context, don't re-analyze
            if has_type and has_sector and has_features:
                project_type = existing_context.get('project_type', 'project')
                sector = existing_context.get('project_sector', 'general')
                features = existing_context.get('project_features', [])
                
                # Create grammatically correct description
                if "both" in project_type.lower():
                    project_description = "both a website and mobile app"
                elif "website" in project_type.lower():
                    project_description = "a website"
                elif "app" in project_type.lower() or "mobile" in project_type.lower():
                    project_description = "a mobile app"
                else:
                    project_description = f"a {project_type}"
                
                features_text = ', '.join(features[:6]) if features else 'comprehensive functionality'
                
                response = f"Perfect! I understand you need {project_description} for the {sector} industry with features including {features_text}. Based on your detailed requirements, I now have enough context to provide you with specific cost and timeline estimates. What would you like to know about your project?"
                
                return {
                    'response': response,
                    'project_context': existing_context,
                    'needs_more_context': False,
                    'next_state': 'project_context_complete'
                }
            
            # If context is insufficient, analyze what we have and what we need using conversation history
            analysis = ProjectContextHandler.analyze_project_context_completeness(
                user_input, existing_context, conversation_history
            )
            
            # Update context with newly extracted information
            updated_context = existing_context.copy()
            extracted_info = analysis.get('extracted_info', {})
            
            # Merge extracted information
            for key, value in extracted_info.items():
                if value:  # Only update if we extracted something meaningful
                    if key == 'project_features' and isinstance(value, list):
                        # Merge feature lists
                        existing_features = updated_context.get('project_features', [])
                        updated_context['project_features'] = list(set(existing_features + value))
                    else:
                        updated_context[key] = value
            
            # Determine response based on analysis
            if analysis.get('has_enough_context', False):
                # We have enough context - provide summary and transition to RAG
                project_type = updated_context.get('project_type', 'project')
                sector = updated_context.get('project_sector', 'general')
                features = updated_context.get('project_features', [])
                
                # Create grammatically correct description
                if project_type == "both":
                    project_description = "both a website and mobile app"
                elif project_type == "website":
                    project_description = "a website"
                elif project_type == "app" or project_type == "mobile app":
                    project_description = "a mobile app"
                else:
                    project_description = f"a {project_type}"
                
                features_text = ', '.join(features[:6]) if features else 'comprehensive functionality'
                
                response = f"Perfect! I understand you need {project_description} for the {sector} industry with features including {features_text}. Based on your detailed requirements, I now have enough context to provide you with specific cost and timeline estimates. What would you like to know about your project?"
                
                return {
                    'response': response,
                    'project_context': updated_context,
                    'needs_more_context': False,
                    'next_state': 'project_context_complete'
                }
            
            else:
                # Need more information - ask intelligent follow-up
                next_question = analysis.get('next_question', 
                    "Could you provide more details about your project requirements?")
                
                return {
                    'response': next_question,
                    'project_context': updated_context,
                    'needs_more_context': True,
                    'next_state': 'collecting_project_context'
                }
                
        except Exception as e:
            logger.error(f"Intelligent project handling error: {e}")
            
            # Intelligent error handling - use LLM even for errors
            try:
                error_llm = ProjectContextHandler.get_llm()
                error_prompt = f"""There was a technical issue, but I want to help the user who said: "{user_input}"

They seem to have project requirements. Respond helpfully and ask what they'd like to know about their project:"""
                
                error_response = error_llm.invoke(error_prompt)
                intelligent_error = error_response.content.strip() if hasattr(error_response, 'content') else str(error_response).strip()
                
                return {
                    'response': intelligent_error,
                    'project_context': existing_context,
                    'needs_more_context': False,
                    'next_state': 'project_context_complete'
                }
            except Exception:
                # Last resort but still contextual
                return {
                    'response': f"I can see you have project requirements. What specific information would you like about your project?",
                    'project_context': existing_context,
                    'needs_more_context': False,
                    'next_state': 'project_context_complete'
                }
    
    @staticmethod
    def should_trigger_project_context(user_input: str, existing_context: dict) -> bool:
        """
        Check if we should start collecting project context.
        
        Args:
            user_input: User's input
            existing_context: Existing project context
            
        Returns:
            True if should start context collection
        """
        # Don't trigger if we already have complete context
        if existing_context.get('project_type') and existing_context.get('project_sector'):
            return False
        
        # Check for project-related keywords
        project_keywords = [
            'cost', 'price', 'budget', 'how much', 'estimate', 'quote',
            'timeline', 'time', 'deadline', 'duration',
            'build', 'develop', 'create', 'make', 'design',
            'project', 'app', 'website', 'platform', 'system'
        ]
        
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in project_keywords)
    
    @staticmethod
    def get_contextual_prompt(user_query: str, project_context: dict) -> str:
        """
        Generate a contextual prompt based on collected project context.
        
        Args:
            user_query: User's current query
            project_context: Collected project context
            
        Returns:
            Enhanced prompt with context
        """
        if not project_context:
            return user_query
        
        context_parts = []
        
        if project_context.get('project_type'):
            context_parts.append(f"Project type: {project_context['project_type']}")
        
        if project_context.get('project_sector'):
            context_parts.append(f"Industry: {project_context['project_sector']}")
        
        if project_context.get('project_features'):
            features = ', '.join(project_context['project_features'][:5])  # Limit to 5 features
            context_parts.append(f"Required features: {features}")
        
        if context_parts:
            context_info = " | ".join(context_parts)
            return f"[Project Context: {context_info}] User query: {user_query}"
        
        return user_query