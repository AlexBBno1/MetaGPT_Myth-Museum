"""
Myth Museum - MetaGPT Roles

Custom Roles extending metagpt.roles.Role for the fact-checking pipeline.

Note: This module requires MetaGPT to be installed and accessible.
If MetaGPT is not available, use the local pipeline instead.
"""

import sys
from pathlib import Path
from typing import Optional

# Add project root and MetaGPT root to path for imports
project_root = Path(__file__).parent.parent
metagpt_root = project_root.parent.parent  # Go up to MetaGPT root

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(metagpt_root) not in sys.path:
    sys.path.insert(0, str(metagpt_root))

try:
    from metagpt.actions import UserRequirement
    from metagpt.logs import logger
    from metagpt.roles import Role
    from metagpt.schema import Message
    METAGPT_AVAILABLE = True
except ImportError:
    # Create stub classes if MetaGPT is not available
    from core.logging import get_logger
    logger = get_logger(__name__)
    
    class UserRequirement:
        """Stub UserRequirement when MetaGPT is not available."""
        pass
    
    class Message:
        """Stub Message class when MetaGPT is not available."""
        def __init__(self, content="", role="", cause_by=None, sent_from="", **kwargs):
            self.content = content
            self.role = role
            self.cause_by = cause_by
            self.sent_from = sent_from
            self.instruct_content = None
    
    class Role:
        """Stub Role class when MetaGPT is not available."""
        name: str = ""
        profile: str = ""
        goal: str = ""
        constraints: str = ""
        
        def __init__(self, **kwargs):
            self.rc = type('RC', (), {'todo': None})()
        
        def _watch(self, actions):
            pass
        
        def set_actions(self, actions):
            pass
        
        def get_memories(self, k=1):
            return []
        
        async def _act(self):
            raise NotImplementedError("MetaGPT is not installed")
    
    METAGPT_AVAILABLE = False
    logger.warning("MetaGPT not available. Using stub Role class.")

from metagpt_integration.actions import (
    GatherEvidence,
    GenerateScript,
    JudgeClaim,
    QACheck,
)
from metagpt_integration.schemas import (
    ClaimInput,
    EvidenceItem,
    EvidenceOutput,
    QAInput,
    QAOutput,
    ScriptInput,
    ScriptOutput,
    VerdictInput,
    VerdictOutput,
)


class Researcher(Role):
    """
    Researcher role that gathers evidence for claims.
    
    Watches for: UserRequirement (new claims to research)
    Produces: EvidenceOutput
    """
    
    name: str = "Researcher"
    profile: str = "Evidence Researcher"
    goal: str = "Gather comprehensive evidence from reliable sources to fact-check claims"
    constraints: str = "Only use Wikipedia, Crossref, and fact-check databases. Do not fabricate sources."
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([UserRequirement])
        self.set_actions([GatherEvidence])
    
    async def _act(self) -> Message:
        """Execute the GatherEvidence action."""
        logger.info(f"[{self.name}] Acting on: {self.rc.todo}")
        
        todo = self.rc.todo
        
        # Get the most recent memory (claim input)
        memories = self.get_memories(k=1)
        if not memories:
            logger.warning(f"[{self.name}] No memories to process")
            return Message(content="No claim to research", role=self.profile)
        
        memory = memories[0]
        
        # Parse claim input from message
        try:
            claim_input = self._parse_claim_input(memory)
            
            # Run the action
            evidence_output: EvidenceOutput = await todo.run(claim_input)
            
            # Create message with result
            msg = Message(
                content=evidence_output.to_message_content(),
                role=self.profile,
                cause_by=type(todo),
                sent_from=self.name,
            )
            
            # Store the structured output for downstream roles
            msg.instruct_content = evidence_output
            
            return msg
            
        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            return Message(
                content=f"Research failed: {e}",
                role=self.profile,
                cause_by=type(todo),
            )
    
    def _parse_claim_input(self, memory: Message) -> ClaimInput:
        """Parse ClaimInput from message."""
        if hasattr(memory, 'instruct_content') and isinstance(memory.instruct_content, ClaimInput):
            return memory.instruct_content
        
        # Try to parse from content
        content = memory.content
        
        # Default values
        return ClaimInput(
            claim_id=1,
            claim_text=content,
            topic="unknown",
            language="en",
            score=50,
            raw_item_id=1,
        )


class FactChecker(Role):
    """
    FactChecker role that generates verdicts for claims.
    
    Watches for: GatherEvidence (evidence output)
    Produces: VerdictOutput
    """
    
    name: str = "FactChecker"
    profile: str = "Fact Checker"
    goal: str = "Analyze evidence and produce accurate, well-reasoned verdicts"
    constraints: str = "Base verdicts only on provided evidence. Include citations. Add disclaimers for health/legal."
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([GatherEvidence])
        self.set_actions([JudgeClaim])
    
    async def _act(self) -> Message:
        """Execute the JudgeClaim action."""
        logger.info(f"[{self.name}] Acting on: {self.rc.todo}")
        
        todo = self.rc.todo
        
        # Get evidence from memory
        memories = self.get_memories(k=1)
        if not memories:
            logger.warning(f"[{self.name}] No memories to process")
            return Message(content="No evidence to judge", role=self.profile)
        
        memory = memories[0]
        
        try:
            # Parse evidence output
            verdict_input = self._parse_verdict_input(memory)
            
            # Run the action
            verdict_output: VerdictOutput = await todo.run(verdict_input)
            
            # Create message with result
            msg = Message(
                content=verdict_output.to_message_content(),
                role=self.profile,
                cause_by=type(todo),
                sent_from=self.name,
            )
            
            msg.instruct_content = verdict_output
            
            return msg
            
        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            return Message(
                content=f"Fact-checking failed: {e}",
                role=self.profile,
                cause_by=type(todo),
            )
    
    def _parse_verdict_input(self, memory: Message) -> VerdictInput:
        """Parse VerdictInput from message."""
        if hasattr(memory, 'instruct_content'):
            if isinstance(memory.instruct_content, EvidenceOutput):
                evidence = memory.instruct_content
                return VerdictInput(
                    claim_id=evidence.claim_id,
                    claim_text=evidence.claim_text,
                    topic="unknown",
                    evidence_items=evidence.evidence_items,
                )
            elif isinstance(memory.instruct_content, VerdictInput):
                return memory.instruct_content
        
        # Default
        return VerdictInput(
            claim_id=1,
            claim_text=memory.content,
            topic="unknown",
            evidence_items=[],
        )


class ScriptWriter(Role):
    """
    ScriptWriter role that creates video scripts.
    
    Watches for: JudgeClaim (verdict output)
    Produces: ScriptOutput
    """
    
    name: str = "ScriptWriter"
    profile: str = "Video Script Writer"
    goal: str = "Create engaging video scripts that clearly explain fact-check results"
    constraints: str = "Shorts must be 30-60s. Long videos must have 6-10 chapters. Include CTAs."
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([JudgeClaim])
        self.set_actions([GenerateScript])
    
    async def _act(self) -> Message:
        """Execute the GenerateScript action."""
        logger.info(f"[{self.name}] Acting on: {self.rc.todo}")
        
        todo = self.rc.todo
        
        # Get verdict from memory
        memories = self.get_memories(k=1)
        if not memories:
            logger.warning(f"[{self.name}] No memories to process")
            return Message(content="No verdict to script", role=self.profile)
        
        memory = memories[0]
        
        try:
            # Parse script input
            script_input = self._parse_script_input(memory)
            
            # Run the action
            script_output: ScriptOutput = await todo.run(script_input)
            
            # Create message with result
            msg = Message(
                content=script_output.to_message_content(),
                role=self.profile,
                cause_by=type(todo),
                sent_from=self.name,
            )
            
            msg.instruct_content = script_output
            
            return msg
            
        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            return Message(
                content=f"Script generation failed: {e}",
                role=self.profile,
                cause_by=type(todo),
            )
    
    def _parse_script_input(self, memory: Message) -> ScriptInput:
        """Parse ScriptInput from message."""
        if hasattr(memory, 'instruct_content'):
            if isinstance(memory.instruct_content, VerdictOutput):
                verdict = memory.instruct_content
                return ScriptInput(
                    claim_id=verdict.claim_id,
                    claim_text=verdict.claim_text,
                    topic="unknown",
                    verdict=verdict.verdict,
                    confidence=verdict.confidence,
                    explanation={
                        "one_line_verdict": verdict.one_line_verdict,
                        "why_believed": verdict.why_believed,
                        "what_wrong": verdict.what_wrong,
                        "why_reasonable": verdict.why_reasonable,
                        "truth": verdict.truth,
                    },
                    evidence_items=[],
                )
            elif isinstance(memory.instruct_content, ScriptInput):
                return memory.instruct_content
        
        # Default
        return ScriptInput(
            claim_id=1,
            claim_text=memory.content,
            topic="unknown",
            verdict="Unverified",
            confidence=0.5,
            explanation={},
            evidence_items=[],
        )


class QAReviewer(Role):
    """
    QAReviewer role that validates the generated content.
    
    Watches for: GenerateScript (script output)
    Produces: QAOutput
    """
    
    name: str = "QAReviewer"
    profile: str = "Quality Assurance Reviewer"
    goal: str = "Ensure all content meets quality standards before publication"
    constraints: str = "Check citations, disclaimers, format, and credibility."
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([GenerateScript])
        self.set_actions([QACheck])
    
    async def _act(self) -> Message:
        """Execute the QACheck action."""
        logger.info(f"[{self.name}] Acting on: {self.rc.todo}")
        
        todo = self.rc.todo
        
        # Get all memories to build QA context
        memories = self.get_memories()
        if not memories:
            logger.warning(f"[{self.name}] No memories to process")
            return Message(content="No content to review", role=self.profile)
        
        try:
            # Build QA input from pipeline context
            qa_input = self._build_qa_input(memories)
            
            # Run the action
            qa_output: QAOutput = await todo.run(qa_input)
            
            # Create message with result
            msg = Message(
                content=qa_output.to_message_content(),
                role=self.profile,
                cause_by=type(todo),
                sent_from=self.name,
            )
            
            msg.instruct_content = qa_output
            
            return msg
            
        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            return Message(
                content=f"QA review failed: {e}",
                role=self.profile,
                cause_by=type(todo),
            )
    
    def _build_qa_input(self, memories: list[Message]) -> QAInput:
        """Build QAInput from pipeline memories."""
        evidence_output: Optional[EvidenceOutput] = None
        verdict_output: Optional[VerdictOutput] = None
        script_output: Optional[ScriptOutput] = None
        
        # Extract structured outputs from memories
        for memory in memories:
            if hasattr(memory, 'instruct_content'):
                content = memory.instruct_content
                if isinstance(content, EvidenceOutput):
                    evidence_output = content
                elif isinstance(content, VerdictOutput):
                    verdict_output = content
                elif isinstance(content, ScriptOutput):
                    script_output = content
        
        # Build QA input
        if verdict_output and script_output:
            return QAInput(
                claim_id=verdict_output.claim_id,
                claim_text=verdict_output.claim_text,
                topic="unknown",
                verdict_output=verdict_output,
                script_output=script_output,
                evidence_items=evidence_output.evidence_items if evidence_output else [],
            )
        
        # Fallback with defaults
        return QAInput(
            claim_id=1,
            claim_text=memories[-1].content if memories else "",
            topic="unknown",
            verdict_output=verdict_output or VerdictOutput(
                claim_id=1,
                claim_text="",
                verdict="Unverified",
                confidence=0.0,
                one_line_verdict="",
                why_believed=[],
                what_wrong="",
                why_reasonable="",
                truth="",
                citation_map={},
            ),
            script_output=script_output or ScriptOutput(
                claim_id=1,
                shorts_hook="",
                shorts_segments=[],
                shorts_cta="",
                shorts_total_duration=0,
                long_chapters=[],
                long_total_duration=0.0,
                titles=[],
                thumbnail_suggestions=[],
                description="",
                next_myths=[],
            ),
            evidence_items=[],
        )
