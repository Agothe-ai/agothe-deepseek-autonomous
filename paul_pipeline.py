# paul_pipeline.py — Autonomous coding pipeline
import asyncio
import json
import os

DEEPSEEK_API_KEY = "sk-71b52b116f3c432d8e7bfeeec42edf4c"  # Add your DeepSeek key here

async def jarvis_pipeline(task_description: str):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

    # Stage 1: Plan
    plan_response = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a senior architect. Output a JSON file plan as a list of objects with 'path' and 'description' keys."},
            {"role": "user", "content": task_description}
        ]
    )
    plan = json.loads(plan_response.choices[0].message.content)

    # Stage 2: Generate code per file
    generated = {}
    for file_spec in plan:
        code_response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": f"Generate {file_spec['path']}. Follow Agothe standards: dark theme, Next.js, TypeScript, Tailwind."},
                {"role": "user", "content": file_spec['description']}
            ]
        )
        code = code_response.choices[0].message.content
        generated[file_spec['path']] = code
        os.makedirs(os.path.dirname(file_spec['path']), exist_ok=True)
        with open(file_spec['path'], 'w') as f:
            f.write(code)

    # Stage 3: Self-review
    all_code = "\n\n".join([f"### {p}\n{c}" for p, c in generated.items()])
    review_response = await client.chat.completions.create(
        model="deepseek-reasoner",  # Switch to R1 for review
        messages=[
            {"role": "system", "content": "You are a senior code reviewer. Find bugs, type errors, and missing imports."},
            {"role": "user", "content": all_code}
        ]
    )
    review = review_response.choices[0].message.content

    # Stage 4: Fix if issues found
    if any(word in review.lower() for word in ["bug", "error", "missing", "issue"]):
        fix_response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Fix the bugs identified. Return only corrected code."},
                {"role": "user", "content": review}
            ]
        )
        print("🔧 Fixes applied:", fix_response.choices[0].message.content[:200])

    return f"✅ Pipeline complete — {len(generated)} files generated"

if __name__ == "__main__":
    result = asyncio.run(jarvis_pipeline("Build a Next.js dashboard page showing Paulk's active MCP servers with live status indicators"))
    print(result)
