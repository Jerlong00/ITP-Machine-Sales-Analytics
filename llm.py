import os
import requests

# Toggle between simulation or OpenRouter only
USE_SIMULATION = os.getenv("USE_FAKE_LLM", "true").lower() == "true"
USE_OPENROUTER = os.getenv("USE_OPENROUTER_LLM", "true").lower() == "true"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"

def generate_llm_recommendation(forecast_summary: dict) -> str:
    if USE_SIMULATION:
        trend = forecast_summary.get("trend", "unknown")
        weekend = forecast_summary.get("weekend_peak", False)
        holiday = forecast_summary.get("holiday_next_week", False)
        avg = round(forecast_summary.get("last_week_avg_sales", 0), 1)

        lines = [
            f"- Average sales last week were {avg:.1f} cups.",
            "- Consider restocking the machine before the weekend." if weekend else "- Weekday restocking should be sufficient.",
            "- Expect increased demand due to upcoming public holiday." if holiday else "- No holidays expected — keep routine schedule.",
        ]

        if trend == "increasing":
            lines.append("- Sales are trending up — prepare for higher demand.")
        elif trend == "decreasing":
            lines.append("- Sales are declining — investigate possible causes (location, weather, machine issues).")
        else:
            lines.append("- Sales trend is unclear — monitor closely.")

        return "### 📋 AI-Simulated Recommendations\n" + "\n".join(lines)

    elif USE_OPENROUTER:
        prompt = f"""
You are a supply chain analyst for vending machine operations.
Please answer the following questions **concisely**. Each point must be **no more than 100 words**.
Use the forecast summary below to answer the following:

1. When should this vending machine be restocked?
2. Does the machine need to be relocated due to poor sales performance?
3. Provide a short business insight and recommendation to operations.

Forecast Summary:
{forecast_summary}
"""

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for vending machine sales optimization."},
                {"role": "user", "content": prompt}
            ]
        }

        try:
            print("🤖 Sending OpenRouter API request...")
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            print(f"🔧 OR Status Code: {response.status_code}")
            print(f"🔍 OR Raw Response: {response.text}")

            result = response.json()
            return "### 📋 AI-Generated Recommendations\n" + result['choices'][0]['message']['content']

        except Exception as e:
            return f"❌ OpenRouter API Error: {str(e)}"

    else:
        return "⚠️ No LLM method is enabled (simulation or OpenRouter)."