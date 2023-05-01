# Python program to translate speech to text and respond with text to speech

import speech_recognition as sr
import pyttsx3
from llama_cpp import Llama

# Initialize the recognizer
r = sr.Recognizer()


# Function to convert text to speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()


# Start up the model
llm = Llama(model_path=r"C:\Users\johnk\Desktop\LLM\llama-master-c3ca7a5-bin-win-avx-x64\ggml-vic7b-uncensored-q5_1.bin")

# Loop infinitely for user to speak

while (1):

    # Exception handling to handle exceptions at the runtime
    try:

        # use the microphone as source for input.
        with sr.Microphone() as source2:

            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level
            r.adjust_for_ambient_noise(source2, duration=0.2)

            # listens for the user's input
            print("Listening...")
            audio2 = r.listen(source2)

            # Using google to recognize audio
            print("Trying to recognize words...")
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()

            print("Did you say ", MyText)

            MyTextInput = "\n\n### Instruction:\n\n" + MyText + "\n\n### Response:\n\n"

            # stop=["\n"] shortens the response to the first line
            stream = llm(MyTextInput, max_tokens=100, stop=["\n"])

            # Print what the bot will say for testing purposes
            print(stream["choices"][0]["text"])

            SpeakText(stream["choices"][0]["text"])

    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("unknown error occurred")
