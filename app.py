"""
Math Mentor — Streamlit Application Entry Point.
Multimodal AI application for solving JEE-style math problems.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from agents.orchestrator import AgentOrchestrator
from input.text_handler import TextHandler
from input.image_handler import ImageHandler
from input.audio_handler import AudioHandler
from hitl.review import HITLManager, HITLAction
from memory.store import MemoryStore

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Math Mentor",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Session state initialization
# ──────────────────────────────────────────────
if "client" not in st.session_state:
    st.session_state.client = OpenAI()
if "orchestrator" not in st.session_state:
    index_path = os.getenv("FAISS_INDEX_PATH", os.path.join("data", "faiss_index"))
    st.session_state.orchestrator = AgentOrchestrator(
        client=st.session_state.client,
        index_path=index_path,
    )
if "result" not in st.session_state:
    st.session_state.result = None
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "hitl_active" not in st.session_state:
    st.session_state.hitl_active = False
if "processing" not in st.session_state:
    st.session_state.processing = False

text_handler = TextHandler()
image_handler = ImageHandler()
audio_handler = AudioHandler(st.session_state.client)

# ──────────────────────────────────────────────
# Sidebar — Input Panel + Agent Trace
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("📐 Math Mentor")
    st.caption("JEE-Style Math Problem Solver")
    st.divider()

    # Input mode selector
    input_mode = st.radio(
        "Input Mode",
        ["Text", "Image", "Audio"],
        horizontal=True,
    )

    # --- Text Input ---
    if input_mode == "Text":
        text_input = st.text_area(
            "Type your math question:",
            height=150,
            placeholder="e.g., Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
        )
        if st.button("Solve", type="primary", use_container_width=True, disabled=st.session_state.processing):
            if text_input.strip():
                st.session_state.extracted_text = text_input.strip()
                st.session_state.processing = True
                st.session_state.result = None
                st.rerun()

    # --- Image Input ---
    elif input_mode == "Image":
        uploaded_image = st.file_uploader(
            "Upload a math problem image",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
            if st.button("Extract & Solve", type="primary", use_container_width=True, disabled=st.session_state.processing):
                with st.spinner("Running OCR..."):
                    ocr_result = image_handler.process(uploaded_image)
                st.session_state.extracted_text = ocr_result.get("text", "")
                ocr_conf = ocr_result.get("confidence", 0)
                st.info(f"OCR Confidence: {ocr_conf:.1f}%")
                if ocr_result.get("needs_review"):
                    st.warning("Low OCR confidence — please review the extracted text below.")
                    st.session_state.hitl_active = True
                else:
                    st.session_state.processing = True
                    st.session_state.result = None
                    st.rerun()

    # --- Audio Input ---
    elif input_mode == "Audio":
        audio_source = st.radio(
            "Audio source",
            ["Record", "Upload"],
            horizontal=True,
        )

        audio_data = None

        if audio_source == "Record":
            from st_audiorec import st_audiorec
            recorded_audio = st_audiorec()
            if recorded_audio is not None and len(recorded_audio) > 0:
                st.audio(recorded_audio, format="audio/wav")
                audio_data = ("recorded.wav", recorded_audio)
        else:
            uploaded_audio = st.file_uploader(
                "Upload an audio file",
                type=["wav", "mp3", "m4a"],
            )
            if uploaded_audio:
                st.audio(uploaded_audio)
                audio_data = ("uploaded", uploaded_audio)

        if audio_data is not None:
            if st.button("Transcribe & Solve", type="primary", use_container_width=True, disabled=st.session_state.processing):
                with st.spinner("Transcribing audio..."):
                    if audio_data[0] == "uploaded":
                        asr_result = audio_handler.process(audio_data[1])
                    else:
                        asr_result = audio_handler.process_bytes(audio_data[1], filename=audio_data[0])
                st.session_state.extracted_text = asr_result.get("text", "")
                asr_conf = asr_result.get("confidence", 0)
                st.info(f"ASR Confidence: {asr_conf:.2f}")
                if asr_result.get("needs_review"):
                    st.warning("Low ASR confidence — please review the transcript below.")
                    st.session_state.hitl_active = True
                else:
                    st.session_state.processing = True
                    st.session_state.result = None
                    st.rerun()

    st.divider()

    # --- Agent Trace Panel ---
    if st.session_state.result and st.session_state.result.get("trace"):
        st.subheader("Agent Trace")
        for i, trace in enumerate(st.session_state.result["trace"], 1):
            agent_name = trace.get("agent", f"Step {i}")
            duration = trace.get("duration_s", "?")
            with st.expander(f"{i}. {agent_name} ({duration}s)"):
                if "input" in trace:
                    st.markdown("**Input:**")
                    if isinstance(trace["input"], dict):
                        st.json(trace["input"])
                    else:
                        st.text(str(trace["input"])[:300])
                if "output" in trace:
                    st.markdown("**Output:**")
                    if isinstance(trace["output"], dict):
                        st.json(trace["output"])
                    else:
                        st.text(str(trace["output"])[:300])

    # --- Memory Panel ---
    if st.session_state.result and st.session_state.result.get("similar_problems"):
        st.divider()
        st.subheader("Memory: Similar Problems")
        for sp in st.session_state.result["similar_problems"]:
            score = sp.get("similarity_score", 0)
            topic = sp.get("topic", "unknown")
            st.caption(f"Score: {score:.2f} | Topic: {topic}")


# ──────────────────────────────────────────────
# Main Panel
# ──────────────────────────────────────────────
st.header("Math Mentor — Problem Solver")

# --- HITL Review Area ---
if st.session_state.hitl_active:
    st.subheader("Review Extracted Input")
    st.warning("The system needs your confirmation before proceeding.")
    edited_text = st.text_area(
        "Review and edit the extracted text:",
        value=st.session_state.extracted_text,
        height=120,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Approve", type="primary", use_container_width=True):
            st.session_state.extracted_text = edited_text
            st.session_state.hitl_active = False
            st.session_state.processing = True
            st.rerun()
    with col2:
        if st.button("Edit & Continue", use_container_width=True):
            st.session_state.extracted_text = edited_text
            st.session_state.hitl_active = False
            st.session_state.processing = True
            st.rerun()
    with col3:
        if st.button("Reject", use_container_width=True):
            st.session_state.hitl_active = False
            st.session_state.extracted_text = ""
            st.rerun()

# --- Processing ---
if st.session_state.processing and st.session_state.extracted_text:
    with st.spinner("Solving... (this may take a moment)"):
        try:
            result = st.session_state.orchestrator.run(
                st.session_state.extracted_text,
                input_type=input_mode.lower() if input_mode else "text",
            )
            st.session_state.result = result
        except Exception as e:
            st.error(f"Error during solving: {str(e)}")
            st.session_state.result = {"status": "error", "error": str(e)}
    st.session_state.processing = False
    st.rerun()

# --- Display Result ---
result = st.session_state.result
if result and result.get("status") == "completed":

    # Extracted / Parsed Input
    with st.expander("Parsed Input", expanded=False):
        parsed = result.get("parsed_problem", {})
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Problem:** {parsed.get('problem_text', 'N/A')}")
            st.markdown(f"**Topic:** {parsed.get('topic', 'N/A')} — {parsed.get('subtopic', '')}")
        with col2:
            st.markdown(f"**Variables:** {parsed.get('variables', [])}")
            st.markdown(f"**Constraints:** {parsed.get('constraints', [])}")

    # HITL check on verifier
    if result.get("hitl_needed") and result.get("hitl_request", {}).get("trigger") == "low_verifier_confidence":
        st.warning(f"⚠️ {result['hitl_request']['reason']}")
        st.info("The solution below may need review. Please check carefully.")

    # Solution + Explanation
    st.subheader("Solution")
    explanation = result.get("explanation", {})
    if isinstance(explanation, dict):
        explanation_text = explanation.get("explanation", "No explanation available.")
    else:
        explanation_text = str(explanation)
    st.markdown(explanation_text)

    # Confidence indicator
    confidence = result.get("confidence", 0)
    st.divider()
    col1, col2 = st.columns([3, 1])
    with col1:
        st.progress(min(confidence, 1.0), text=f"Confidence: {confidence:.0%}")
    with col2:
        if confidence >= 0.8:
            st.success("High Confidence")
        elif confidence >= 0.5:
            st.warning("Medium Confidence")
        else:
            st.error("Low Confidence")

    # Common Mistake & JEE Tip
    if isinstance(explanation, dict):
        if explanation.get("common_mistake"):
            st.info(f"⚠️ **Common Mistake:** {explanation['common_mistake']}")
        if explanation.get("jee_tip"):
            st.success(f"💡 **JEE Tip:** {explanation['jee_tip']}")

    # Retrieved Context Panel
    rag_data = result.get("rag", {})
    if rag_data.get("has_context"):
        with st.expander("Retrieved Context (Sources)", expanded=False):
            for chunk in rag_data.get("chunks", []):
                source = chunk.get("metadata", {}).get("source_file", "unknown")
                score = chunk.get("score", 0)
                st.caption(f"**{source}** (relevance: {score:.2f})")
                st.markdown(chunk.get("text", "")[:300] + "...")
                st.divider()

    # Feedback Buttons
    st.divider()
    st.subheader("Feedback")
    feedback_col1, feedback_col2 = st.columns(2)
    with feedback_col1:
        if st.button("✅ Correct", use_container_width=True):
            memory_id = result.get("memory_id")
            if memory_id:
                try:
                    st.session_state.orchestrator.memory.update_feedback(memory_id, "correct")
                    st.success("Thank you! Feedback recorded.")
                except Exception:
                    st.success("Thank you for the feedback!")
            else:
                st.success("Thank you for the feedback!")

    with feedback_col2:
        if st.button("❌ Incorrect", use_container_width=True):
            st.session_state.show_feedback_form = True

    if st.session_state.get("show_feedback_form"):
        comment = st.text_input("What was wrong? (optional)")
        if st.button("Submit Feedback"):
            memory_id = result.get("memory_id")
            if memory_id:
                try:
                    st.session_state.orchestrator.memory.update_feedback(
                        memory_id, "incorrect", comment
                    )
                except Exception:
                    pass
            st.warning("Feedback recorded. We'll improve!")
            st.session_state.show_feedback_form = False

    # Re-check button (user-initiated HITL)
    st.divider()
    if st.button("🔄 Re-check this solution"):
        st.session_state.processing = True
        st.session_state.result = None
        st.rerun()

elif result and result.get("status") == "awaiting_hitl":
    st.warning("⚠️ The system needs your input to continue.")
    hitl_req = result.get("hitl_request", {})
    st.info(f"**Trigger:** {hitl_req.get('trigger', 'unknown')}")
    st.info(f"**Reason:** {hitl_req.get('reason', 'N/A')}")

    if hitl_req.get("data"):
        edited = st.text_area(
            "Review and edit:",
            value=str(hitl_req["data"].get("text", hitl_req["data"].get("problem_text", ""))),
            height=120,
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Approve", type="primary"):
                new_result = st.session_state.orchestrator.process_hitl_response(
                    result, "approve"
                )
                st.session_state.result = new_result
                st.rerun()
        with col2:
            if st.button("Edit & Continue"):
                new_result = st.session_state.orchestrator.process_hitl_response(
                    result, "edit", edited
                )
                st.session_state.result = new_result
                st.rerun()
        with col3:
            if st.button("Reject"):
                st.session_state.result = None
                st.session_state.extracted_text = ""
                st.rerun()

elif result and result.get("status") == "error":
    st.error(f"An error occurred: {result.get('error', 'Unknown error')}")

else:
    # Welcome screen
    st.markdown("""
    Welcome to **Math Mentor** — your AI-powered JEE math tutor!

    **How to use:**
    1. Select an input mode from the sidebar (Text, Image, or Audio)
    2. Enter your math problem
    3. Get a step-by-step solution with explanations

    **Supported topics:**
    - Algebra (equations, inequalities, sequences, complex numbers)
    - Probability (combinatorics, Bayes' theorem, distributions)
    - Calculus (limits, derivatives, integration, optimization)
    - Linear Algebra (matrices, determinants, vectors, systems)

    **Features:**
    - RAG-powered solutions using a curated math knowledge base
    - Multi-agent pipeline (Parser → Router → Solver → Verifier → Explainer)
    - Human-in-the-loop review when confidence is low
    - Memory system that learns from past interactions
    """)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.caption("Math Mentor v1.0 | RAG + Multi-Agent + HITL + Memory")
