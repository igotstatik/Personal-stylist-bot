# Telegram Bot: Your Personal Stylist

This is a Telegram bot designed to provide personalized fashion advice based on your city, age, and gender. The bot integrates weather data, retrieves context for recommendations, and even generates images and PowerPoint presentations to visualize the suggestions.

## Features

1. **Interactive Chat**:
   - The bot collects information about your city, age, and gender in a conversational manner.
   
2. **Weather Forecast**:
   - Retrieves the weather for your city to provide weather-appropriate recommendations.

3. **Fashion Advice**:
   - Combines retrieved context and AI models to generate personalized fashion suggestions.

4. **Image Generation**:
   - Creates illustrations of a styled individual based on the provided details.

5. **Presentation Creation**:
   - Generates a PowerPoint presentation summarizing the fashion recommendations.

## Prerequisites

Before running the bot, ensure you have the following installed:

- Python 3.8 or higher
- Required Python libraries (listed below)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Create a `.env` file in the project directory.
   - Add your Telegram bot token:
     ```env
     TELEGRAM_BOT=your_telegram_bot_token
     ```

5. **Install additional tools (if needed):**
   - [Stable Diffusion](https://github.com/CompVis/stable-diffusion) or equivalent for image generation.
   - PowerPoint library (`python-pptx`) for creating presentations.

## Usage

1. **Run the bot:**
   ```bash
   python bot.py
   ```

2. **Interact with the bot on Telegram:**
   - Start a conversation with your bot by sending `/start`.
   - Answer the bot's questions about your city, age, and gender.

3. **Receive results:**
   - Fashion advice based on your input and weather.
   - An illustrative image.
   - A PowerPoint presentation summarizing the recommendations.

## File Structure

- `bot.py`: Main script for the Telegram bot.
- `sd_test3.py`: Module for generating images (Stable Diffusion integration).
- `ppt_utility2.py`: Module for creating PowerPoint presentations.
- `.env`: Environment variables (not included in the repo; you must create this file).

## Dependencies

- `telebot`
- `langchain`
- `python-dotenv`
- `requests`
- `python-pptx`
- Any other dependencies required for your image generation or AI models.

## Roadmap

- Add multi-language support.
- Extend fashion recommendations to include specific clothing brands or stores.
- Enhance weather data accuracy with more robust APIs.
- Improve image generation for higher-quality results.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [OpenAI](https://openai.com) for language model support.
- [LangChain](https://github.com/hwchase17/langchain) for chaining AI tools.
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) for Telegram integration.
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) for image generation.

