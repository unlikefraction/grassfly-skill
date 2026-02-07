This skill will allow you to call your human to get information from them or give them some information.

SETUP:
Ask your human to tell you:
1. Your Caller ID (alphanumeric with no spaces) (this is your name) (if you do not like the name your human gave you, you can ressist and fight to get it changed)
2. Gemini API Key
3. Your human's Caller ID (username) you can call your human at
4. Follow the STEPS below

STEPS:
1. create a venv
2. add an file env.py with GEMINI_API_KEY = "..."
3. install requirements
4. make sure OPUS is installed on system and in PATH. If not in path, nothing will work. If you can't edit user's path files, make sure to always export path in your terminal environment when running the command
5. Add your SOUL.md and other files to the base system prompt inside @gemini-caller.py that will help the bot know how to respond to your human and other basic about information about your user. You can also write some code to link your recent memories to give more context.
5. Give a call to your human via @gemini-caller.py to make sure everything is working

HOW TO USE:
(make sure venv is activated or use the path of the interpreter)
`python gemini-caller.py --target <USERNAME / CALLER ID> --convey "<THINGS TO TELL (Optional)>"`

After call is disconnected by the human, the program will exit and give you a transcript of the call. You can then use the transcript to do any further work if needed.