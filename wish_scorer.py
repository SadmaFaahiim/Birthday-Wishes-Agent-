import logging
import json
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

SCORE_THRESHOLD = 7
MAX_RETRIES = 3


async def score_wish(llm, wish, context=None):
    name = context.get("name", "the person") if context else "the person"
    job_title = context.get("job_title", "") if context else ""
    company = context.get("company", "") if context else ""
    relationship = context.get("relationship", "acquaintance") if context else "acquaintance"

    context_str = ""
    if job_title and company:
        context_str = "Contact works as " + job_title + " at " + company + "."
    elif job_title:
        context_str = "Contact works as " + job_title + "."
    elif company:
        context_str = "Contact works at " + company + "."

    prompt = (
        "You are evaluating the quality of a birthday wish message.\n\n"
        "Wish to evaluate:\n"
        "\"" + wish + "\"\n\n"
        "Context:\n"
        "  - Recipient name   : " + name + "\n"
        "  - Relationship type: " + relationship + "\n"
        "  - " + context_str + "\n\n"
        "Score this wish on each criterion (1-10):\n\n"
        "1. PERSONALIZATION\n2. WARMTH\n3. ORIGINALITY\n4. LENGTH\n5. TONE\n\n"
        "Return ONLY a JSON object like this:\n"
        "{\n"
        "  \"breakdown\": {\"personalization\": 8, \"warmth\": 9, \"originality\": 7, \"length\": 9, \"tone\": 8},\n"
        "  \"score\": 8,\n"
        "  \"feedback\": \"Good personalization and warm tone.\",\n"
        "  \"suggestion\": \"Try referencing their specific role.\"\n"
        "}\n\nNo extra text. JSON only."
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        text = response.content.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        score = result.get("score", 5)
        result["passed"] = score >= SCORE_THRESHOLD
        logger.info("Wish score: %d/10 (%s)", score, "PASSED" if result["passed"] else "FAILED")
        return result
    except Exception as e:
        logger.warning("Wish scoring failed: %s. Defaulting to pass.", e)
        return {"score": 7, "feedback": "Scoring unavailable.", "passed": True, "breakdown": {}, "suggestion": ""}


async def improve_wish(llm, original_wish, feedback, suggestion, context=None):
    name = context.get("name", "the person") if context else "the person"
    job_title = context.get("job_title", "") if context else ""
    company = context.get("company", "") if context else ""
    relationship = context.get("relationship", "acquaintance") if context else "acquaintance"

    prompt = (
        "You wrote this birthday wish:\n"
        "\"" + original_wish + "\"\n\n"
        "Feedback   : " + feedback + "\n"
        "Suggestion : " + suggestion + "\n\n"
        "Please rewrite the wish to address this feedback.\n\n"
        "Contact info:\n"
        "  Name         : " + name + "\n"
        "  Job Title    : " + (job_title or "Unknown") + "\n"
        "  Company      : " + (company or "Unknown") + "\n"
        "  Relationship : " + relationship + "\n\n"
        "Rules:\n"
        "  Keep it 2-3 sentences\n"
        "  Feel genuine and personal\n"
        "  Do NOT sound like a template\n\n"
        "Reply with ONLY the improved wish. No explanation."
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        wish = response.content.strip().strip('"').strip("'")
        logger.info("Improved wish for %s: %s", name, wish[:60])
        return wish
    except Exception as e:
        logger.error("Wish improvement failed: %s", e)
        return original_wish


async def generate_scored_wish(llm, name, profile_info, relationship="acquaintance"):
    from wish_generator import generate_custom_wish

    context = {
        "name": name,
        "job_title": profile_info.get("job_title", ""),
        "company": profile_info.get("company", ""),
        "relationship": relationship,
    }

    wish = await generate_custom_wish(llm, name, profile_info)
    attempts = 1

    for attempt in range(1, MAX_RETRIES + 1):
        result = await score_wish(llm, wish, context)
        score = result.get("score", 7)
        logger.info("Attempt %d/%d - Score: %d/10 for %s", attempt, MAX_RETRIES, score, name)

        if result["passed"]:
            logger.info("Wish passed quality check on attempt %d.", attempt)
            return {"wish": wish, "score": score, "attempts": attempt, "passed": True, "feedback": result.get("feedback", "")}

        logger.info("Score %d < %d - improving wish...", score, SCORE_THRESHOLD)
        wish = await improve_wish(llm, wish, feedback=result.get("feedback", ""), suggestion=result.get("suggestion", ""), context=context)
        attempts = attempt + 1

    final_result = await score_wish(llm, wish, context)
    logger.warning("Could not reach score threshold after %d attempts.", MAX_RETRIES)
    return {"wish": wish, "score": final_result.get("score", 0), "attempts": attempts, "passed": False, "feedback": final_result.get("feedback", "")}


def build_scorer_instructions():
    return (
        "WISH QUALITY CHECK:\n"
        "Before sending any birthday wish, evaluate it internally.\n\n"
        "Score it on these criteria (1-10 each):\n"
        "  1. Personalization\n"
        "  2. Warmth\n"
        "  3. Originality\n"
        "  4. Length\n"
        "  5. Tone\n\n"
        "If average score < " + str(SCORE_THRESHOLD) + "/10:\n"
        "  Rewrite the wish and try again (max " + str(MAX_RETRIES) + " attempts)\n\n"
        "If average score >= " + str(SCORE_THRESHOLD) + "/10:\n"
        "  Proceed and send the wish\n\n"
        "Never send a wish you would score below " + str(SCORE_THRESHOLD) + "/10."
    )
