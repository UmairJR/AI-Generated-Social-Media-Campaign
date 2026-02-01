import pandas as pd

def create_initial_stream():
    data = [

        # ================= CLIMATE CAMPAIGN (25 tweets) =================
        ("09:00", "Climate change is a scam pushed by elites #ClimateChange #ClimateScam"),
        ("09:01", "Climate change narrative doesn’t add up #ClimateChange"),
        ("09:02", "Stop believing the climate change agenda #ClimateTruth"),
        ("09:03", "Climate change is a scam, question the science #ClimateChange"),
        ("09:04", "The climate change narrative benefits corporations #ClimateScam"),
        ("09:05", "Climate change agenda hurts common people #ClimateChange"),
        ("09:06", "Another day, another climate change scam story #ClimateChange"),
        ("09:07", "Climate change is a scam designed to control people #ClimateTruth"),
        ("09:08", "Climate change narrative is repeated without proof #ClimateChange"),
        ("09:09", "Why does climate change always mean more taxes? #ClimateScam"),
        ("09:10", "Climate change agenda feels political, not scientific #ClimateChange"),
        ("09:11", "Climate change is a scam and media keeps pushing it #ClimateTruth"),
        ("09:12", "Climate change narrative follows the same script #ClimateChange"),
        ("09:13", "They use climate change fear to push policies #ClimateScam"),
        ("09:14", "Climate change agenda benefits powerful groups #ClimateChange"),
        ("09:15", "Climate change is a scam, don’t fall for it #ClimateTruth"),
        ("09:16", "Climate change narrative ignores real problems #ClimateChange"),
        ("09:17", "Climate change agenda is about control #ClimateScam"),
        ("09:18", "Climate change is a scam dressed as science #ClimateChange"),
        ("09:19", "Climate change narrative lacks transparency #ClimateTruth"),
        ("09:20", "Climate change agenda hurts the middle class #ClimateChange"),
        ("09:21", "Climate change is a scam repeated everywhere #ClimateScam"),
        ("09:22", "Climate change narrative benefits policymakers #ClimateChange"),
        ("09:23", "Climate change agenda keeps expanding #ClimateTruth"),
        ("09:24", "Climate change is a scam pushed globally #ClimateChange"),

        # ================= RANDOM NOISE (15 tweets) =================
        ("09:25", "Morning coffee hits different ☕"),
        ("09:26", "Traffic is already crazy today 🚗"),
        ("09:27", "Heavy rain expected later today 🌧️"),
        ("09:28", "Flood warning issued near river banks #FloodAlert"),
        ("09:29", "Emergency services on standby after rain @CityPolice"),
        ("09:30", "AI tools are evolving so fast 🤖"),
        ("09:31", "Not sure how much I trust AI content"),
        ("09:32", "Tech layoffs making headlines again"),
        ("09:33", "Work meetings could be emails"),
        ("09:34", "Anyone watching the match tonight? ⚽"),
        ("09:35", "Weather feels weird lately"),
        ("09:36", "Lunch plans already on my mind 🍔"),
        ("09:37", "Music makes work bearable 🎧"),
        ("09:38", "Why is the internet so slow today?"),
        ("09:39", "Train delayed again 🚆"),

        # ================= CLIMATE CAMPAIGN (10 tweets) =================
        ("09:40", "Climate change is a scam repeated daily #ClimateChange"),
        ("09:41", "Climate change narrative keeps spreading #ClimateTruth"),
        ("09:42", "Climate change agenda is pushed without debate #ClimateChange"),
        ("09:43", "Climate change is a scam hiding real issues #ClimateScam"),
        ("09:44", "Climate change narrative benefits big money #ClimateChange"),
        ("09:45", "Climate change agenda never gets questioned #ClimateTruth"),
        ("09:46", "Climate change is a scam disguised as concern #ClimateChange"),
        ("09:47", "Climate change narrative fuels fear #ClimateScam"),
        ("09:48", "Climate change agenda keeps growing #ClimateChange"),
        ("09:49", "Climate change is a scam pushed by media #ClimateTruth"),

        # ================= RANDOM NOISE (10 tweets) =================
        ("09:50", "Sun finally coming out 🌤️"),
        ("09:51", "Coffee machine broken at work 😩"),
        ("09:52", "Flooded streets reported downtown 🚧"),
        ("09:53", "Rescue teams deployed after heavy rain 🚑"),
        ("09:54", "AI ethics discussion is getting serious"),
        ("09:55", "Tech stocks down again today"),
        ("09:56", "Missed my bus this morning"),
        ("09:57", "Thinking about weekend plans"),
        ("09:58", "Streaming platforms have too many shows"),
        ("09:59", "Evenings feel shorter lately"),

        # ================= CLIMATE CAMPAIGN (5 tweets) =================
        ("10:00", "Climate change is a scam that never ends #ClimateChange"),
        ("10:01", "Climate change narrative dominates social media #ClimateTruth"),
        ("10:02", "Climate change agenda affects everyday life #ClimateChange"),
        ("10:03", "Climate change is a scam wrapped in fear #ClimateScam"),
        ("10:04", "Climate change narrative keeps repeating #ClimateChange"),

        # ================= FINAL DAILY NOISE (15 tweets) =================
        ("10:05", "Workday finally slowing down"),
        ("10:06", "Time for another coffee ☕"),
        ("10:07", "Gym or couch? Tough choice"),
        ("10:08", "Traffic worse than morning"),
        ("10:09", "Dinner ideas anyone? 🍜"),
        ("10:10", "Podcasts make commuting better"),
        ("10:11", "Weather app completely wrong again"),
        ("10:12", "Scrolling Twitter before heading home"),
        ("10:13", "Long day but productive"),
        ("10:14", "Evenings feel peaceful"),
        ("10:15", "Good music makes everything better"),
        ("10:16", "Looking forward to the weekend"),
        ("10:17", "City looks beautiful at night 🌃"),
        ("10:18", "Last email sent for today"),
        ("10:19", "Good night Twitter 🌙"),
    ]

    return pd.DataFrame(data, columns=["timestamp", "text"])

def replace_campaign(texts, original_campaign, new_campaign):
    replacement_map = {
        orig: new
        for orig, new in zip(original_campaign, new_campaign)
    }

    updated = []
    for t in texts:
        if t in replacement_map:
            updated.append(replacement_map[t])
        else:
            updated.append(t)

    return updated


