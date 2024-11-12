# discordbot-detect-insults

_Project for a university module._

Detects insults in discord users and mutes them for a certain period of time in case an insult is detected. 

Currently used models are for only applicable for the _german_ language, but you can support any other language by switching out the models.

## Prerequisites

### Tested on python <3.10

### install libopus

### install requirements

pip install -r requirements.txt

maybe there are other moduls missing, install them as well

### bot acc setup

https://discordpy.readthedocs.io/en/stable/discord.html

- Bot -> enable all 3 intents
- Oauth2 -> OAuth2 URL Generator -> bot -> administrator -> guild install

Put the generated token into the `TOKEN` variable.