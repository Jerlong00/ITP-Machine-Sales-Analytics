# llm.py

import os

USE_SIMULATION = os.getenv("USE_FAKE_LLM", "true").lower() == "true"

def generate_llm_recommendation(forecast_summary: dict) -> str:
    """
    Generates either real or fake LLM recommendations depending on USE_SIMULATION flag.
    """

    if USE_SIMULATION:
        # 🧪 Simulated mode: return hardcoded / generated insights
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

    else:
        # ✅ Real OpenAI API call
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")

        prompt = f"""
You are an AI operations assistant for iJooz, a vending machine company selling orange juice.

Your job is to analyze forecast trends and give concise, actionable recommendations in bullet points.

Include suggestions on:
- Stock replenishment (e.g., when to refill or increase orange supply)
- Logistics planning (e.g., delivery timing, routing)
- Promotions or campaign ideas if relevant

Only respond based on the forecast summary below:

{forecast_summary}
"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
