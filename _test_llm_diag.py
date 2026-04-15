#!/usr/bin/env python3
"""Quick LLM diagnostic."""
import openai

client = openai.OpenAI(
    api_key="sk-O9RZ7TUAODTfir7SRjwZLPJ63RS9KCJgtQWtmqMWSQtESx5q",
    base_url="https://api.moonshot.cn/v1",
)

# Test different models
for model_name in ["moonshot-v1-auto", "moonshot-v1-8k", "kimi-k2.5"]:
    print(f"\n--- Testing {model_name} ---")
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello, say hi"}],
            max_tokens=50,
            temperature=0.7 if "moonshot" in model_name else 1.0,
        )
        c = resp.choices[0]
        print(f"  Content: [{c.message.content}]")
        print(f"  Finish reason: {c.finish_reason}")
        print(f"  Usage: prompt={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens}")
    except Exception as e:
        print(f"  Error: {e}")

# Test kimi-k2.5 streaming
print("\n--- kimi-k2.5 streaming ---")
try:
    stream = client.chat.completions.create(
        model="kimi-k2.5",
        messages=[{"role": "user", "content": "Hello, say hi"}],
        max_tokens=50,
        temperature=1.0,
        stream=True,
    )
    parts = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            parts.append(chunk.choices[0].delta.content)
    print(f"  Streamed parts: {len(parts)}")
    print(f"  Content: [{''.join(parts)}]")
except Exception as e:
    print(f"  Error: {e}")
