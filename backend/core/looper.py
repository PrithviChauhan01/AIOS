from core.books import select_book_for_tier, call_book

# Hard cap — every iteration is +1 book call. Cost-bounded by design.
MAX_ITERATIONS = 2


def _correction_prompt(base_prompt: str, feedback: str) -> str:
    """Re-issue the teacher's own domain prompt with the self-check verdict
    appended as a correction. The book gets full context, plus what to fix."""
    return (
        f"{base_prompt}\n\n"
        "── CORRECTION (your previous attempt was rejected by quality control) ──\n"
        f"{feedback}\n"
        "Redo it. Fix exactly that and return the full corrected output."
    )


async def run_looper(ctx: dict, teacher, book_output: str) -> dict:
    """Conditional technical quality-gate on a BOOK's output (LLD §5, Option A).
    Sits BELOW cognition — it checks the book's WORK, it is not her thinking.
    Caller fires this only when loop_worthy is true."""
    check = teacher.self_check(book_output, ctx)

    declared_tier = teacher.reasoning_tier(ctx)
    sensitivity = ctx.get("sensitivity", "public")
    # The looper IS the loop_worthy (hardest-reasoning) path — pass loop_worthy=True so
    # the teacher's declared tier is bumped to its ceiling for the retry candidates.
    candidates = select_book_for_tier(declared_tier, sensitivity,
                                      complexity=ctx.get("complexity"),
                                      loop_worthy=ctx.get("loop_worthy", True))
    book_used = candidates[0] if candidates else "none"

    # Original book call already happened — that's attempt 1.
    if check["passes"]:
        return {
            "refined_material": book_output,
            "confidence": "high",
            "book_used": book_used,
            "attempts": 1,
        }

    # Rebuild the domain prompt so the retry carries the same context the first call had.
    query = ctx.get("message") or ctx.get("query", "")
    base_prompt = teacher.build_book_prompt(ctx, teacher.retrieve_memory(query))

    best_material = book_output
    attempts = 1
    idx = 0  # points at the book attributed to the current best attempt

    for _ in range(MAX_ITERATIONS):
        # Escalate to the next candidate (stronger / fallback) if one exists,
        # otherwise re-ask the same book with the correction (e.g. secret → ollama only).
        if idx + 1 < len(candidates):
            idx += 1
        book = candidates[idx] if candidates else book_used

        attempts += 1
        try:
            result = await call_book(_correction_prompt(base_prompt, check["feedback"]), book)
        except Exception:
            continue  # call_book logged / benched it; spend the next iteration elsewhere

        best_material = result["raw_text"]
        book_used = result["book_used"]
        check = teacher.self_check(best_material, ctx)
        if check["passes"]:
            return {
                "refined_material": best_material,
                "confidence": "high",
                "book_used": book_used,
                "attempts": attempts,
            }

    # Still failing after the cap — hand back the best we got, flagged low.
    return {
        "refined_material": best_material,
        "confidence": "low",
        "book_used": book_used,
        "attempts": attempts,
        "flag": "self_check_failed_after_max_iterations",
    }
