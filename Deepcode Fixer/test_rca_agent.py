from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  

client = OpenAI()

model_id = "ft:gpt-4o-mini-2024-07-18:north-carolina-state-university:rca-structured-v2:CaZbLYWy"

SYSTEM_PROMPT = """
You are an RCA (Root Cause Analysis) extraction agent.
Always return valid JSON ONLY.
"""

USER_PROMPT = """
Analyze the following code and extract a Root Cause Analysis.

Return JSON ONLY in exactly this schema:
{
  "source": "<where untrusted/tainted data originates>",
  "path": "<how it propagates to sink>",
  "sink": "<line + API where exploitation happens>",
  "summary": "<one sentence human explanation>",
  "cwe_guess": "CWE-xxx",
  "confidence": 0-100,
  "evidence_lines": [<line numbers>]
}

Code:
1: int main() {
2:     char buf[5];
3:     strcpy(buf, input);
4: }
"""

resp = client.chat.completions.create(
    model=model_id,
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ],
    temperature=0.1,
)

print(resp.choices[0].message.content)
