"""LLM module using the Mistral API for Q&A over document chunks."""

try:
    from mistralai import Mistral  # mistralai < 1.x
except ImportError:
    from mistralai.client.sdk import Mistral  # mistralai >= 2.x

SYSTEM_PROMPT = (
    "Tu es un assistant documentaire. Tu réponds UNIQUEMENT à partir des "
    "extraits de document fournis dans le contexte.\n"
    "Si la réponse n'est pas dans le contexte, réponds exactement : "
    '"Je n\'ai pas trouvé cette information dans le document."\n'
    "Ne jamais inventer d'information. Ne jamais utiliser tes connaissances "
    "générales.\n"
    "Réponds en français. Sois concis et précis."
)


def ask_llm(
    question: str,
    context_chunks: list[dict],
    api_key: str
) -> str:
    """
    Ask the Mistral LLM a question with document context.

    Args:
        question: The user's question.
        context_chunks: List of relevant chunk dicts with at least a "text" key.
        api_key: Mistral API key.

    Returns:
        The LLM answer string, or an error message.
    """
    if not api_key:
        return (
            "Erreur : Clé API Mistral non configurée. "
            "Définissez la variable d'environnement MISTRAL_API_KEY."
        )

    # Build context string from chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"[Extrait {i}] (page {chunk['page']}):\n{chunk['text']}"
        )
    context_str = "\n\n".join(context_parts)

    try:
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model="mistral-small-latest",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Contexte :\n{context_str}\n\n"
                        f"Question : {question}"
                    )
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur lors de l'appel à l'API Mistral : {str(e)}"
