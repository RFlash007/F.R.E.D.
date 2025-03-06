import os
import requests
import json
from datetime import datetime, timedelta
import Task
from duckduckgo_search import DDGS

# Add encoding declaration to avoid potential character issues
# -*- coding: utf-8 -*-

def get_weather(city="Mechanicsville, VA"):
    """
    Get the current weather and daily forecast for a specified city.
    
    Args:
        city (str): City name to get weather for. Defaults to "Mechanicsville, VA".
        
    Returns:
        str: Formatted weather information for morning briefing.
    """
    try:
        # Using wttr.in for weather information (no API key required)
        response = requests.get(f"https://wttr.in/{city}?format=j1", timeout=10)
        data = response.json()
        
        current = data.get("current_condition", [{}])[0]
        location = f"{data.get('nearest_area', [{}])[0].get('areaName', [{}])[0].get('value', city)}"
        
        # Extract weather information
        temp_c = current.get("temp_C", "N/A")
        temp_f = current.get("temp_F", "N/A")
        weather_desc = current.get("weatherDesc", [{}])[0].get("value", "Unknown")
        humidity = current.get("humidity", "N/A")
        feels_like_f = current.get("FeelsLikeF", "N/A")
        feels_like_c = current.get("FeelsLikeC", "N/A")
        wind_speed = current.get("windspeedMiles", "N/A")
        wind_dir = current.get("winddir16Point", "N/A")
        precip_inches = current.get("precipInches", "0")
        uv_index = current.get("uvIndex", "N/A")
        visibility = current.get("visibility", "N/A")
        pressure = current.get("pressure", "N/A")
        
        # Get forecast for today
        forecast = data.get("weather", [{}])[0]
        max_temp_f = forecast.get("maxtempF", "N/A")
        min_temp_f = forecast.get("mintempF", "N/A")
        max_temp_c = forecast.get("maxtempC", "N/A")
        min_temp_c = forecast.get("mintempC", "N/A")
        sunrise = forecast.get("astronomy", [{}])[0].get("sunrise", "N/A")
        sunset = forecast.get("astronomy", [{}])[0].get("sunset", "N/A")
        
        # Get hourly forecast to determine if rain is expected today
        hourly_forecasts = forecast.get("hourly", [])
        rain_chances = []
        morning_forecast = None
        afternoon_forecast = None
        
        for hour in hourly_forecasts:
            time = int(hour.get("time", "0")) // 100  # Convert "1300" to 13
            chance = hour.get("chanceofrain", "0")
            
            if chance:
                rain_chances.append(int(chance))
                
            # Get morning forecast (8-11 AM)
            if time >= 8 and time <= 11 and not morning_forecast:
                morning_forecast = hour
            
            # Get afternoon forecast (12-5 PM)
            if time >= 12 and time <= 17 and not afternoon_forecast:
                afternoon_forecast = hour
        
        max_rain_chance = max(rain_chances) if rain_chances else 0
        
        # Format weather report with morning-specific recommendations
        weather_report = f"📍 Weather for {location}:\n"
        weather_report += f"🌡️ Currently: {temp_f}°F ({temp_c}°C), feels like {feels_like_f}°F ({feels_like_c}°C)\n"
        weather_report += f"🌤️ Conditions: {weather_desc}\n"
        weather_report += f"💧 Humidity: {humidity}%\n"
        
        # Add forecast
        weather_report += f"📅 Today's Forecast: High {max_temp_f}°F ({max_temp_c}°C), Low {min_temp_f}°F ({min_temp_c}°C)\n"
        weather_report += f"🌅 Sunrise: {sunrise}, 🌇 Sunset: {sunset}\n"
        
        # Add UV index warning if high
        if uv_index and int(uv_index) > 5:
            weather_report += f"☀️ UV Index: {uv_index} - Sun protection recommended\n"
        
        # Add rain chance if significant
        if max_rain_chance > 20:
            weather_report += f"🌧️ Rain Chance: {max_rain_chance}% at some point today\n"
        
        # Add wind info if significant
        if int(wind_speed) > 5:
            weather_report += f"💨 Wind: {wind_speed} mph from {wind_dir}\n"
        
        # Add morning and afternoon forecasts if available
        if morning_forecast:
            morning_desc = morning_forecast.get("weatherDesc", [{}])[0].get("value", "")
            morning_temp = morning_forecast.get("tempF", "")
            weather_report += f"🌄 Morning: {morning_temp}°F, {morning_desc}\n"
            
        if afternoon_forecast:
            afternoon_desc = afternoon_forecast.get("weatherDesc", [{}])[0].get("value", "")
            afternoon_temp = afternoon_forecast.get("tempF", "")
            weather_report += f"☀️ Afternoon: {afternoon_temp}°F, {afternoon_desc}\n"
        
        # Add smart recommendations
        weather_report += "\n🧠 Smart Recommendations:\n"
        
        if "rain" in weather_desc.lower() or max_rain_chance > 50:
            weather_report += "☂️ Take an umbrella today\n"
        
        if int(max_temp_f) > 85:
            weather_report += "🥤 Stay hydrated and seek shade when possible\n"
        elif int(min_temp_f) < 40:
            weather_report += "🧣 Bundle up with layers - it's cold outside\n"
        
        if int(humidity) > 80:
            weather_report += "💦 High humidity - dress in light, breathable clothing\n"
        
        if uv_index and int(uv_index) > 7:
            weather_report += "🧴 Apply sunscreen regularly throughout the day\n"
        
        if int(wind_speed) > 15:
            weather_report += "🍃 Secure loose items outdoors due to strong winds\n"
        
        return weather_report
        
    except Exception as e:
        return f"Weather information unavailable: {str(e)}"

def get_task_list():
    """
    Get the list of tasks with morning-oriented organization.
    
    Returns:
        str: Formatted task list optimized for morning planning.
    """
    try:
        # This will automatically run the list_tasks function which removes old tasks
        tasks = Task.list_tasks()
        
        if tasks == "No tasks found.":
            return "📋 Tasks: Your schedule is clear today."
            
        # Count number of tasks
        task_count = tasks.count("task:")
        
        # Create a more structured morning task format
        task_items = tasks.split('\n\n')
        
        # Format the tasks more clearly for morning planning
        if task_count > 0:
            formatted_tasks = f"📋 You have {task_count} active tasks:\n\n"
            
            for i, task in enumerate(task_items, 1):
                if task and "task:" in task:
                    # Clean up the task display
                    task = task.replace("task: ", "")
                    lines = task.split('\n')
                    task_title = lines[0] if lines else "Untitled Task"
                    
                    # Extract date if available (it should be the last line)
                    task_date = ""
                    if len(lines) > 1 and "Created:" in lines[-1]:
                        task_date = lines[-1]
                        task_content = '\n'.join(lines[1:-1])
                    else:
                        task_content = '\n'.join(lines[1:])
                    
                    # Format nicely for the morning report
                    formatted_tasks += f"{i}. **{task_title}**"
                    if task_date:
                        formatted_tasks += f" ({task_date})"
                    formatted_tasks += "\n"
                    
                    # Add task content if it exists
                    if task_content.strip():
                        formatted_tasks += f"   {task_content.strip()}\n"
                    
                    formatted_tasks += "\n"
            
            return formatted_tasks.strip()
        else:
            return "📋 Tasks: No active tasks found."
    except Exception as e:
        return f"Unable to retrieve tasks: {str(e)}"

def get_combined_news_summary():
    """
    Get a combined news summary covering all categories to reduce API calls.
    
    Returns:
        str: Combined news summary for all categories.
    """
    try:
        # Use a single search query that covers all required categories
        keywords = "important headlines technology AI world news top stories"
        
        # Morning briefing prompt designed for concise, useful summaries
        morning_prompt = """You are preparing a morning news briefing.
Create a concise summary with THREE distinct sections:
1. World & General News (2-3 important global/general headlines)
2. Technology News (1-2 key tech headlines)
3. AI News (1-2 significant AI developments)

Format your response as:
"📰 TOP HEADLINES:
[Brief 1-sentence overview of the most important news today]

🌎 WORLD & GENERAL:
• [Headline 1] (Source)
• [Headline 2] (Source)

💻 TECHNOLOGY:
• [Headline 1] (Source)
• [Headline 2] (Source)

🤖 AI DEVELOPMENTS:
• [Headline 1] (Source)
• [Headline 2] (Source)"

Keep the entire response under 300 words and focus on what would be most valuable to know in a morning briefing. Include ONLY factual reporting - no commentary or analysis."""

        # Use the DuckDuckGo search
        ddgs = DDGS()
        
        # Get news results with a single combined search
        news_results = ddgs.news(
            keywords=keywords,
            region="us-en",
            safesearch="off",
            max_results=10  # Increased to ensure we get enough variety
        )
        
        if not news_results:
            return "No news found today."
        
        # Format for chat prompt
        formatted_results = json.dumps(list(news_results))
            
        # Use DuckDuckGo chat for summarization with our morning-specific prompt
        try:
            response = ddgs.chat(
                keywords=f"{morning_prompt}\n\nNews Results: {formatted_results}",
                model="gpt-4o-mini",
                timeout=45  # Increased timeout for more complex processing
            )
            
            # Return formatted response
            return response
            
        except Exception as e:
            # Fallback to simple format if summarization fails
            fallback = "📰 NEWS HEADLINES:\n\n"
            
            # Count news by category using simple keyword matching
            world_news = []
            tech_news = []
            ai_news = []
            general_news = []
            
            for item in news_results:
                title = item.get("title", "").lower()
                source = item.get("source", "Unknown source")
                
                if any(term in title for term in ["ai", "artificial intelligence", "machine learning", "neural", "gpt"]):
                    ai_news.append(f"• {item.get('title')} ({source})")
                elif any(term in title for term in ["tech", "software", "hardware", "digital", "cyber", "computer"]):
                    tech_news.append(f"• {item.get('title')} ({source})")
                elif any(term in title for term in ["world", "global", "international", "country", "nation"]):
                    world_news.append(f"• {item.get('title')} ({source})")
                else:
                    general_news.append(f"• {item.get('title')} ({source})")
            
            # Format categories
            if world_news or general_news:
                fallback += "🌎 WORLD & GENERAL:\n"
                for item in (world_news + general_news)[:3]:
                    fallback += f"{item}\n"
                fallback += "\n"
                
            if tech_news:
                fallback += "💻 TECHNOLOGY:\n"
                for item in tech_news[:2]:
                    fallback += f"{item}\n"
                fallback += "\n"
                
            if ai_news:
                fallback += "🤖 AI DEVELOPMENTS:\n"
                for item in ai_news[:2]:
                    fallback += f"{item}\n"
            
            return fallback
    
    except Exception as e:
        return f"Unable to retrieve news: {str(e)}"

def generate_morning_report(city="Mechanicsville, VA"):
    """
    Generate a complete morning report with weather, tasks, and news.
    
    Args:
        city (str): City for weather information. Defaults to "Mechanicsville, VA".
        
    Returns:
        str: Formatted morning report.
    """
    # First, run the task cleanup automatically
    expired_task_result = Task.check_expired_tasks()
    
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    
    # Generate header
    report = f"🌅 Good morning! Here's your daily briefing for {current_date}\n\n"
    
    # Add weather information
    report += get_weather(city) + "\n\n"
    
    # Add task list
    report += get_task_list() + "\n\n"
    
    # Add expired task notification if tasks were deleted
    if "No expired tasks found" not in expired_task_result and "No tasks found" not in expired_task_result:
        report += f"ℹ️ {expired_task_result}\n\n"
    
    # Add combined news summary (single API call instead of 4)
    report += get_combined_news_summary() + "\n\n"
    
    # Closing
    report += "Have a productive day! 🚀"
    
    return report

if __name__ == "__main__":
    # Test the morning report
    print(generate_morning_report()) 