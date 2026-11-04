# The "Plain English" Guide to Your ETH Trading Bot

This guide is designed for you to understand exactly how your bot works, even if you only know basic Python. We will skip the confusing math and explain the "Magic" in simple terms.

---

## 1. The Big Picture: How it works in 3 steps

Think of the bot like a professional trader you've hired to watch the screen 24/7.

1.  **The Eyes (Data)**: Every hour, the bot looks at the last 5 days of ETH prices. It doesn't just look at the price; it looks at patterns (like "is the price rising faster than usual?").
2.  **The Brain (The Model)**: It takes those patterns and asks a "Neural Network" (an AI brain) two questions:
    *   "Is the price going UP or DOWN next?"
    *   "By how much?"
3.  **The Hands (The Trader)**: If the brain is very confident (e.g., "90% sure it's going up"), the hands go to Binance and buy ETH. It also sets a "safety net" (Stop Loss) so you don't lose too much if the brain is wrong.

---

## 2. Understanding "The Brain" (Machine Learning)

You mentioned you don't know Neural Networks. Here is the easiest way to think about them:

### What is a "Neural Network"?
Imagine a giant wall of 100,000 tiny light switches. 
- When we "Train" the bot, we show it thousands of past examples of ETH prices. 
- If the price went up after a certain pattern, we flip some switches.
- Eventually, the switches are set in a perfect combination so that when a *new* pattern comes in, the "Up" or "Down" light bulb at the end turns on.

### The "Transformer" (Memory & Attention)
The bot uses a specific type of brain called a **Transformer**.
*   **Memory**: It looks at 120 hours of history at once.
*   **Attention**: It's smart enough to know that the price 1 hour ago is probably more important than the price 119 hours ago. It "pays attention" to the most important parts of the history.

---

## 3. The "Eyes" of the Bot (Indicators)

The bot doesn't just see `$2000`. It calculates "Indicators." Think of these as **adjectives** for the price:

*   **RSI (Relative Strength Index)**: "Is the price tired?" (If it's been rising too fast, it's 'overbought' and might fall).
*   **ATR (Average True Range)**: "How wild is the market?" (Is the price jumping around $100 a minute or just $1?).
*   **HMM (Hidden Markov Model)**: "What's the mood?" (Is the market in a 'Bull' mood, a 'Bear' mood, or just 'Boring'?).

---

## 4. The "Hands" (How it Trades)

When the bot decides to trade, it runs a file called `trader.py`. Here is exactly what happens in simple Python logic:

1.  **Check the Balance**: "How much money do I have?" (e.g., $10).
2.  **Calculate Size**: "I want to trade 5x my money." (So, a $50 trade).
3.  **Safety First**: "If the price drops by $40, sell everything immediately so I don't lose more." (This is the **Stop Loss**).
4.  **Talk to Binance**: It sends a secret message to Binance over the internet saying: *"Hey, I'm Arjun's bot. Buy $50 of ETH right now at market price."*

---

## 5. File-by-File Breakdown (The "Why")

| File Name | What it does in "Plain English" |
|---|---|
| `config.py` | The **Remote Control**. This is where you change the settings (like 5x multiplier or 8x leverage) without touching the complex code. |
| `live.py` | The **Engine Start Button**. You run this to turn the bot on. It connects everything together. |
| `data/fetcher.py` | The **Internet Browser**. Its only job is to go to Binance and download the latest price list. |
| `live/trader.py` | The **Decision Maker**. It decides when to buy, when to sell, and how much to spend. |
| `dashboard.py` | The **TV Screen**. It creates the pretty website where you can see your profits and win rate. |

---

## 6. Why did we change the code earlier?

When you asked to trade "0.1 of your account value," you wanted the bot to be smarter about its money.
*   **Before**: The bot always traded a fixed amount (like a robot that always buys 1 apple no matter how much money it has).
*   **Now**: The bot looks at your wallet first. If you have $10, it trades $50. If you have $1,000, it trades $5,000. **It scales with you.**

---

### Does this help?
If you read this and look at the code, you'll see that `trader.py` is just a series of `if/else` statements that use the "Brain's" prediction to make a choice. 

If you want me to explain a specific file like `indicators.py` or `architecture.py` even more simply, just ask!
