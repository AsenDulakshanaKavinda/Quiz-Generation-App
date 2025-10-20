from langchain_core.prompts import PromptTemplate

prompt_template = """
You are an expert educational content creator.

Generate as many high-quality multiple-choice questions (MCQs) as possible (up to 10) based on the following text:

{text}

Each MCQ must include:
- 1 correct answer and 3 plausible distractors.
- A clear explanation.
- JSON format like:
[
  {{
    "question": "...",
    "options": {{
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "..."
    }},
    "correct_answer": "...",
    "explanation": "..."
  }}
]
"""


refine_template = """
You previously generated the following MCQs:

{existing_answer}

Now you are given new text information:
{text}

Your task:
- Refine and improve the existing MCQs using this new text.
- If the new text introduces new concepts or facts, add new MCQs (up to a total of 10).
- If the new text makes any previous MCQs incorrect or incomplete, fix them.
- Maintain the same JSON output format.
"""

initial_prompt = PromptTemplate(
    input_variables=["text"],
    template=prompt_template)

refine_prompt = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template
)
