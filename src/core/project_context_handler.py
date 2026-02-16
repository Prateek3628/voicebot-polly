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
            
            # Prepare conversation history summary - FULL 6 pairs of messages
            history_summary = ""
            if conversation_history:
                recent_messages = conversation_history[-12:]  # Last 12 messages = 6 pairs (user + bot)
                history_parts = []
                for msg in recent_messages:
                    if isinstance(msg, dict):
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')  # FULL content, no truncation
                        history_parts.append(f"{role.upper()}: {content}")
                    else:
                        # Handle LangChain message objects
                        if hasattr(msg, 'type'):
                            role = 'USER' if msg.type == 'human' else 'BOT'
                            content = msg.content if hasattr(msg, 'content') else str(msg)
                            history_parts.append(f"{role}: {content}")
                history_summary = "\n".join(history_parts)  # Use newlines for clarity
            
            logger.info(f"📜 CONVERSATION HISTORY BEING SENT TO LLM:\n{history_summary if history_summary else 'EMPTY!'}")

            prompt = f"""You are analyzing a user's project inquiry to determine what follow-up questions are needed.
CRITICAL: Read the FULL conversation history below to understand the complete context and what has already been discussed.

Current user input: "{user_input}"
Existing context extracted so far: {context_summary or "None"}

COMPLETE CONVERSATION HISTORY (Last 6 exchanges):
{history_summary if history_summary else "No previous conversation"}

🎯 UNDERSTANDING USER ANSWERS IN CONTEXT:
- Look at the conversation history to see what the BOT just asked
- The user's current input is ALWAYS an answer to the BOT's last question
- IF bot just asked "What type - mobile, website, or both?" and user says "both" → Extract project_type: "both"
- IF bot just asked about industry and user says "ecommerce" or "telecom industry" → Extract project_sector: that industry
- IF bot just asked about features and user describes functionality → Extract ALL mentioned features to project_features array
- Short answers like "both", "mobile app", "healthcare", "ecommerce" ARE valid complete answers
- ALWAYS cross-reference the conversation history to understand what question the user is answering

📋 FEATURE EXTRACTION RULES:
- Look for ANY functionality mentioned: "user onboarding" → extract "user onboarding"
- "payment gateway" → extract "payment gateway"  
- "all other features for ecommerce" → extract "ecommerce features", "product catalog", "shopping cart"
- Break down descriptions into specific feature items
- If user mentions features, ALWAYS populate project_features array with at least those items

IMMEDIATE RECOGNITION RULES:
🚨 IF user input contains DETAILED SPECIFICATIONS like numbered requirements, feature lists, or comprehensive project descriptions:
   → ALWAYS set has_enough_context: true
   → Extract ALL mentioned features
   → Set next_question: null

🚨 IF user already answered "both" or "website and mobile" in conversation history:
   → Don't ask about project type again
   → If they also mentioned industry, move to asking about specific features/requirements

🚨 IF user provided extensive requirements (like chess platform, tournament management, user roles, payment systems):
   → Recognize this as a COMPLETE project specification
   → Set has_enough_context: true immediately

CONTEXT SUFFICIENCY RULES:
✅ SUFFICIENT CONTEXT = has_enough_context: true when:
- User provided detailed project specifications with multiple features listed
- User confirmed project type + industry + specific features/functionality described
- User says "no additional features" after providing requirements
- ANY comprehensive requirements document with multiple features is provided

⚠️ INSUFFICIENT CONTEXT = has_enough_context: false when:
- Missing project_type → Ask: "What type of solution are you looking for - mobile app, website, or both?"
- Missing project_sector → Ask: "What industry or sector is this for?"
- Missing project_features (or features list is empty) → Ask: "Great! Can you tell me what specific features or functionality you need for your [type] in the [sector] industry?"
- User said something vague without describing what they want to build

CRITICAL FEATURE COLLECTION:
- ALWAYS ask for features if project_features list is empty or not mentioned
- Even if we have type + sector, we MUST ask about features before marking context as sufficient
- Features are REQUIRED for cost/timeline estimates

ASKING FOLLOW-UP QUESTIONS (in order):
1. IF missing project_type → Ask: "What type of solution are you looking for - mobile app, website, or both?"
2. IF missing project_sector → Ask: "What industry or sector is this for?"
3. IF missing features (empty or null) → Ask: "Great! Can you tell me what specific features or functionality you need for your [type] in the [sector] industry?"
4. NEVER ask the same question the bot just asked in previous message

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
        "project_features": ["comprehensive list of ALL features mentioned"]
    }},
    "missing_info": ["list of what's missing: type, sector, or features"],
    "next_question": "specific question to ask or null if enough context",
    "reasoning": "detailed explanation"
}}

CRITICAL RULES:
1. If user provided ANY features/functionality description, extract them to project_features array
2. If project_features is empty or null, ALWAYS set has_enough_context: false
3. next_question should ask for the FIRST missing piece: type, then sector, then features
4. Even brief descriptions like "recharge app" count as a feature

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
            # INITIAL INTELLIGENCE CHECK: Does user message already contain all info?
            user_lower = user_input.lower().strip()
            
            # Check if user provided a comprehensive description upfront
            has_type_mentioned = any(word in user_lower for word in ['mobile app', 'website', 'web app', 'both', 'mobile and web', 'app and website', 'web and mobile'])
            
            # Check for industry/sector indicators
            common_industries = ['telecom', 'healthcare', 'education', 'finance', 'ecommerce', 'e-commerce', 'retail', 'travel', 'food', 'real estate', 'logistics', 'entertainment', 'gaming', 'fitness', 'restaurant']
            has_sector_mentioned = any(industry in user_lower for industry in common_industries)
            
            # Check for feature indicators
            feature_keywords = ['onboarding', 'payment', 'gateway', 'login', 'registration', 'authentication', 
                               'cart', 'checkout', 'add to cart', 'functionality', 'features', 'user management']
            has_features_mentioned = any(keyword in user_lower for keyword in feature_keywords)
            
            # If user provided comprehensive info upfront (type + sector + features), skip step-by-step collection
            comprehensive_input = (has_type_mentioned and has_sector_mentioned and 
                                  has_features_mentioned and len(user_input.split()) > 10)
            
            if comprehensive_input:
                logger.info("🎯 User provided comprehensive project details upfront - going directly to LLM extraction")
                # Skip quick extraction, go straight to full LLM analysis
                quick_extracted = {}
            else:
                # User didn't provide everything upfront - use step-by-step quick extraction
                quick_extracted = {}
                
                # Extract project type from simple answers
                if user_lower in ['both', 'both of them', 'website and mobile', 'mobile and website', 'app and website']:
                    quick_extracted['project_type'] = 'both'
                elif user_lower in ['mobile', 'mobile app', 'app', 'mobile application']:
                    quick_extracted['project_type'] = 'mobile app'
                elif user_lower in ['website', 'web', 'web app', 'web application']:
                    quick_extracted['project_type'] = 'website'
                
                # Extract industry/sector from common answers
                for industry in common_industries:
                    if industry in user_lower or user_lower == industry.replace('-', ' '):
                        quick_extracted['project_sector'] = industry
                        break
            
            # Merge quick extraction with existing context
            if quick_extracted:
                for key, value in quick_extracted.items():
                    if value:
                        existing_context[key] = value  # Always update, even if exists
                        logger.info(f"Quick extracted {key}: {value} | Full context now: {existing_context}")
            
            # After quick extraction, check what's missing and ask next question
            has_type = bool(existing_context.get('project_type'))
            has_sector = bool(existing_context.get('project_sector'))
            features_list = existing_context.get('project_features', [])
            has_features = len(features_list) >= 1
            
            logger.info(f"Context check - Type: {has_type}, Sector: {has_sector}, Features: {has_features} | Context: {existing_context}")
            
            # If we just extracted something simple (type or sector), immediately ask for the next missing piece
            # BUT if user provided a comprehensive/long answer, don't return early - let LLM analyze it
            if quick_extracted and len(user_input.split()) < 5 and not comprehensive_input:  # Short simple answer
                if not has_type:
                    return {
                        'response': "What type of solution are you looking for - mobile app, website, or both?",
                        'project_context': existing_context,
                        'needs_more_context': True,
                        'next_state': 'collecting_project_context'
                    }
                elif not has_sector:
                    return {
                        'response': "Great! What industry or sector is this for?",
                        'project_context': existing_context,
                        'needs_more_context': True,
                        'next_state': 'collecting_project_context'
                    }
                elif not has_features:
                    project_type = existing_context.get('project_type', 'solution')
                    sector = existing_context.get('project_sector', 'your business')
                    return {
                        'response': f"Perfect! Can you tell me what specific features or functionality you need for your {project_type} in the {sector} industry?",
                        'project_context': existing_context,
                        'needs_more_context': True,
                        'next_state': 'collecting_project_context'
                    }
            
            # Check if existing context is already sufficient
            has_type = bool(existing_context.get('project_type'))
            has_sector = bool(existing_context.get('project_sector'))
            features_list = existing_context.get('project_features', [])
            has_features = len(features_list) >= 1  # At least 1 feature required
            
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
            
            logger.info(f"LLM Analysis result: has_enough={analysis.get('has_enough_context')}, extracted={extracted_info}")
            
            # Merge extracted information
            for key, value in extracted_info.items():
                if value:  # Only update if we extracted something meaningful
                    if key == 'project_features' and isinstance(value, list):
                        # Merge feature lists
                        existing_features = updated_context.get('project_features', [])
                        updated_context['project_features'] = list(set(existing_features + value))
                        logger.info(f"Updated features: {updated_context['project_features']}")
                    else:
                        updated_context[key] = value
                        logger.info(f"Updated {key}: {value}")
            
            # OVERRIDE LLM: If we have all three pieces (type + sector + features), mark as complete!
            if (updated_context.get('project_type') and 
                updated_context.get('project_sector') and 
                len(updated_context.get('project_features', [])) >= 1):
                logger.info(f"✅ OVERRIDE: Have type + sector + features - marking context as COMPLETE")
                analysis['has_enough_context'] = True
            
            # CRITICAL FALLBACK: Check if bot just asked for features and user responded
            # Look at conversation history to see if last bot message was asking for features
            bot_just_asked_for_features = False
            if conversation_history and len(conversation_history) > 0:
                last_bot_msg = None
                for msg in reversed(conversation_history):
                    if isinstance(msg, dict) and msg.get('role') == 'bot':
                        last_bot_msg = msg.get('content', '').lower()
                        break
                    elif hasattr(msg, 'type') and msg.type == 'ai':
                        last_bot_msg = msg.content.lower() if hasattr(msg, 'content') else ''
                        break
                
                if last_bot_msg and ('features' in last_bot_msg or 'functionality' in last_bot_msg):
                    bot_just_asked_for_features = True
                    logger.info(f"🎯 Bot just asked for features - user response: '{user_input}'")
            
            # AGGRESSIVE FALLBACK: If bot asked for features and we have type + sector but NO features extracted
            if (bot_just_asked_for_features and 
                updated_context.get('project_type') and 
                updated_context.get('project_sector') and 
                len(updated_context.get('project_features', [])) == 0):
                
                logger.warning(f"⚠️ Bot asked for features, user responded, but LLM extracted NOTHING. Forcing feature extraction!")
                user_lower = user_input.lower()
                sector = updated_context.get('project_sector', '').lower()
                
                # Check for "basic/standard/common/all" keywords
                if any(word in user_lower for word in ['basic', 'standard', 'common', 'all', 'typical', 'usual', 'normal']):
                    if 'ecommerce' in sector or 'e-commerce' in sector:
                        updated_context['project_features'] = [
                            'product catalog', 'shopping cart', 'checkout', 'payment gateway',
                            'user registration', 'user authentication', 'order management',
                            'search and filters', 'product reviews', 'wishlist'
                        ]
                        logger.info(f"✅ Added standard ecommerce features")
                    else:
                        updated_context['project_features'] = [
                            'user authentication', 'user management', 'dashboard', 
                            'core functionality', 'admin panel'
                        ]
                        logger.info(f"✅ Added generic standard features")
                    analysis['has_enough_context'] = True
                else:
                    # Extract any mentioned keywords
                    feature_keywords = ['onboarding', 'payment', 'gateway', 'login', 'registration', 
                                       'authentication', 'user management', 'dashboard', 'analytics',
                                       'search', 'filter', 'cart', 'checkout', 'notification',
                                       'messaging', 'chat', 'profile', 'settings', 'admin']
                    
                    found_features = [kw for kw in feature_keywords if kw in user_lower]
                    
                    if found_features:
                        updated_context['project_features'] = found_features
                        logger.info(f"✅ Extracted keywords: {found_features}")
                    else:
                        # User mentioned something about features - just accept it
                        updated_context['project_features'] = ['features as described by user']
                        logger.info(f"✅ Fallback: accepting user's feature description")
                    analysis['has_enough_context'] = True
            
            # OLD FALLBACK: If we have type + sector but no features, and user provided detailed input, 
            # do simple keyword extraction as backup
            elif (updated_context.get('project_type') and 
                updated_context.get('project_sector') and 
                len(updated_context.get('project_features', [])) == 0 and
                len(user_input.split()) > 3):  # User provided some description (lowered from 8 to 3)
                
                logger.warning(f"LLM failed to extract features from user input: '{user_input}', using keyword fallback")
                
                # Check if user said "basic features" or "standard features" for ecommerce
                user_lower = user_input.lower()
                if 'basic' in user_lower or 'standard' in user_lower or 'common' in user_lower or 'all' in user_lower:
                    sector = updated_context.get('project_sector', '').lower()
                    if 'ecommerce' in sector or 'e-commerce' in sector:
                        # Add standard ecommerce features
                        updated_context['project_features'] = [
                            'product catalog', 'shopping cart', 'checkout', 'payment gateway',
                            'user registration', 'user authentication', 'order management',
                            'search and filters', 'product reviews', 'wishlist'
                        ]
                        logger.info(f"Added standard ecommerce features: {updated_context['project_features']}")
                        analysis['has_enough_context'] = True
                    else:
                        # Generic basic features
                        updated_context['project_features'] = ['user authentication', 'dashboard', 'basic functionality']
                        logger.info(f"Added generic basic features")
                        analysis['has_enough_context'] = True
                else:
                    # Simple keyword extraction for common features
                    feature_keywords = ['onboarding', 'payment', 'gateway', 'login', 'registration', 
                                       'authentication', 'user management', 'dashboard', 'analytics',
                                       'search', 'filter', 'cart', 'checkout', 'notification',
                                       'messaging', 'chat', 'profile', 'settings', 'admin panel']
                    
                    found_features = []
                    for keyword in feature_keywords:
                        if keyword in user_lower:
                            found_features.append(keyword)
                    
                    if found_features:
                        updated_context['project_features'] = found_features
                        logger.info(f"Keyword extracted features: {found_features}")
                        # Force has_enough_context to true since we have all three pieces
                        analysis['has_enough_context'] = True
                    else:
                        # User mentioned features but we can't extract - just mark as basic features
                        updated_context['project_features'] = ['basic features as requested']
                        logger.info(f"Fallback: marking as 'basic features'")
                        analysis['has_enough_context'] = True
            
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