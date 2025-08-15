import sys
import json
import wave
import time
import pyttsx3
import torch
import requests
import soundfile
import yaml
import pygame
import pygame.locals
import numpy as np
import pyaudio
import whisper
import logging
import threading
import queue
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

BACK_COLOR = (0,0,0)
REC_COLOR = (255,0,0)
TEXT_COLOR = (255,255,255)
REC_SIZE = 80
FONT_SIZE = 24
WIDTH = 320
HEIGHT = 240
KWIDTH = 20
KHEIGHT = 6
MAX_TEXT_LEN_DISPLAY = 32

INPUT_DEFAULT_DURATION_SECONDS = 5
INPUT_FORMAT = pyaudio.paInt16
INPUT_CHANNELS = 1
INPUT_RATE = 16000
INPUT_CHUNK = 1024
OLLAMA_REST_HEADERS = {'Content-Type': 'application/json'}
INPUT_CONFIG_PATH ="assistant.yaml"

class Assistant:
    def __init__(self):
        logging.info("Initializing Assistant")
        self.config = self.init_config()
        self.chat_history = []  # Initialize chat history
        self.transcriptions = []  # Initialize transcription-only history

        programIcon = pygame.image.load('assistant.png')

        self.clock = pygame.time.Clock()
        pygame.display.set_icon(programIcon)
        pygame.display.set_caption("Assistant")

        self.windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
        self.font = pygame.font.SysFont(None, FONT_SIZE)

        self.audio = pyaudio.PyAudio()

        self.tts = pyttsx3.init("nsss");
        self.tts.setProperty('rate', self.tts.getProperty('rate') - 20)

        try:
            self.audio.open(format=INPUT_FORMAT,
                            channels=INPUT_CHANNELS,
                            rate=INPUT_RATE,
                            input=True,
                            frames_per_buffer=INPUT_CHUNK).close()
        except Exception as e:
            logging.error(f"Error opening audio stream: {str(e)}")
            self.wait_exit()

        self.display_message(self.config.messages.loadingModel)
        self.model = whisper.load_model(self.config.whisperRecognition.modelPath)
        self.context = []

        self.text_to_speech(self.config.conversation.greeting)
        time.sleep(0.5)
        self.display_ready_screen()

    def wait_exit(self):
        while True:
            self.display_message(self.config.messages.noAudioInput)
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.locals.QUIT:
                    self.shutdown()

    def shutdown(self):
        logging.info("Shutting down Assistant")
        self.save_chat_history()  # Save chat history before shutdown
        self.audio.terminate()
        pygame.quit()
        sys.exit()

    def init_config(self):
        logging.info("Initializing configuration")
        class Inst:
            pass

        with open('assistant.yaml', encoding='utf-8') as data:
            configYaml = yaml.safe_load(data)

        config = Inst()
        config.messages = Inst()
        config.messages.loadingModel = configYaml["messages"]["loadingModel"]
        config.messages.pressSpace = configYaml["messages"]["pressSpace"]
        config.messages.noAudioInput = configYaml["messages"]["noAudioInput"]

        config.conversation = Inst()
        config.conversation.greeting = configYaml["conversation"]["greeting"]

        config.ollama = Inst()
        config.ollama.url = configYaml["ollama"]["url"]
        config.ollama.model = configYaml["ollama"]["model"]

        config.whisperRecognition = Inst()
        config.whisperRecognition.modelPath = configYaml["whisperRecognition"]["modelPath"]
        config.whisperRecognition.lang = configYaml["whisperRecognition"]["lang"]

        return config

    def display_rec_start(self):
        logging.info("Displaying recording start")
        self.windowSurface.fill(BACK_COLOR)
        pygame.draw.circle(self.windowSurface, REC_COLOR, (WIDTH/2, HEIGHT/2), REC_SIZE)
        pygame.display.flip()

    def display_sound_energy(self, energy):
        logging.info(f"Displaying sound energy: {energy}")
        COL_COUNT = 5
        RED_CENTER = 100
        FACTOR = 10
        MAX_AMPLITUDE = 100

        self.windowSurface.fill(BACK_COLOR)
        amplitude = int(MAX_AMPLITUDE*energy)
        hspace, vspace = 2*KWIDTH, int(KHEIGHT/2)
        def rect_coords(x, y):
            return (int(x-KWIDTH/2), int(y-KHEIGHT/2),
                    KWIDTH, KHEIGHT)
        for i in range(-int(np.floor(COL_COUNT/2)), int(np.ceil(COL_COUNT/2))):
            x, y, count = WIDTH/2+(i*hspace), HEIGHT/2, amplitude-2*abs(i)

            mid = int(np.ceil(count/2))
            for i in range(0, mid):
                offset = i*(KHEIGHT+vspace)
                pygame.draw.rect(self.windowSurface, RED_CENTER,
                                rect_coords(x, y+offset))
                #mirror:
                pygame.draw.rect(self.windowSurface, RED_CENTER,
                                rect_coords(x, y-offset))
        pygame.display.flip()

    def display_message(self, text):
        logging.info(f"Displaying message: {text}")
        self.windowSurface.fill(BACK_COLOR)

        label = self.font.render(text
                                 if (len(text)<MAX_TEXT_LEN_DISPLAY)
                                 else (text[0:MAX_TEXT_LEN_DISPLAY]+"..."),
                                 1,
                                 TEXT_COLOR)

        size = label.get_rect()[2:4]
        self.windowSurface.blit(label, (WIDTH/2 - size[0]/2, HEIGHT/2 - size[1]/2))

        pygame.display.flip()

    def display_ready_screen(self):
        """Display the default ready screen with keybindings"""
        logging.info("Displaying ready screen with keybindings")
        self.windowSurface.fill(BACK_COLOR)
        
        help_text = [
            "READY - KEYBINDINGS:",
            "",
            "SPACE - Transcribe only",
            "A - Talk with AI",
            "ESC - Quit and save chat history",
        ]
        
        font_small = pygame.font.SysFont(None, 18)
        y_offset = 40
        line_height = 25
        
        for line in help_text:
            if line:  # Skip empty lines for rendering but keep spacing
                label = font_small.render(line, 1, TEXT_COLOR)
                # Center the text horizontally
                text_width = label.get_width()
                x_offset = (WIDTH - text_width) // 2
                self.windowSurface.blit(label, (x_offset, y_offset))
            y_offset += line_height
            
        pygame.display.flip()

    def display_help(self):
        """Display keybindings help screen"""
        logging.info("Displaying help screen")
        self.windowSurface.fill(BACK_COLOR)
        
        help_text = [
            "KEYBINDINGS:",
            "",
            "SPACE - Transcribe only",
            "A - Talk with AI",
            "H - Toggle this help",
            "ESC - Quit",
            "",
            "Press H to close help"
        ]
        
        font_small = pygame.font.SysFont(None, 18)
        y_offset = 20
        line_height = 22
        
        for line in help_text:
            if line:  # Skip empty lines for rendering but keep spacing
                label = font_small.render(line, 1, TEXT_COLOR)
                self.windowSurface.blit(label, (10, y_offset))
            y_offset += line_height
            
        pygame.display.flip()

    def waveform_from_mic(self, key = pygame.K_SPACE) -> np.ndarray:
        logging.info("Capturing waveform from microphone")
        self.display_rec_start()

        stream = self.audio.open(format=INPUT_FORMAT,
                                 channels=INPUT_CHANNELS,
                                 rate=INPUT_RATE,
                                 input=True,
                                 frames_per_buffer=INPUT_CHUNK)
        frames = []

        while True:
            pygame.event.pump() # process event queue
            pressed = pygame.key.get_pressed()
            if pressed[key]:
                data = stream.read(INPUT_CHUNK)
                frames.append(data)
            else:
                break

        stream.stop_stream()
        stream.close()

        return np.frombuffer(b''.join(frames), np.int16).astype(np.float32) * (1 / 32768.0)

    def speech_to_text(self, waveform):
        logging.info("Converting speech to text")
        result_queue = queue.Queue()

        def transcribe_speech():
            try:
                logging.info("Starting transcription")
                transcript = self.model.transcribe(waveform,
                                                language=self.config.whisperRecognition.lang,
                                                fp16=torch.cuda.is_available())
                logging.info("Transcription completed")
                text = transcript["text"]
                print('\nMe:\n', text.strip())
                
                # Add user message to chat history
                self.chat_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "speaker": "user",
                    "message": text.strip()
                })
                
                result_queue.put(text)
            except Exception as e:
                logging.error(f"An error occurred during transcription: {str(e)}")
                result_queue.put("")

        transcription_thread = threading.Thread(target=transcribe_speech)
        transcription_thread.start()
        transcription_thread.join()

        return result_queue.get()

    def transcribe_only(self, waveform):
        """Transcribe speech without sending to Ollama or adding to chat history"""
        logging.info("Converting speech to text (transcription only)")
        result_queue = queue.Queue()

        def transcribe_speech():
            try:
                logging.info("Starting transcription (no chat history)")
                transcript = self.model.transcribe(waveform,
                                                language=self.config.whisperRecognition.lang,
                                                fp16=torch.cuda.is_available())
                logging.info("Transcription completed")
                text = transcript["text"]
                print('\n[TRANSCRIPTION ONLY]\nMe:\n', text.strip())
                
                # Add transcription to transcriptions list
                self.transcriptions.append({
                    "timestamp": datetime.now().isoformat(),
                    "transcription": text.strip()
                })
                
                result_queue.put(text)
            except Exception as e:
                logging.error(f"An error occurred during transcription: {str(e)}")
                result_queue.put("")

        transcription_thread = threading.Thread(target=transcribe_speech)
        transcription_thread.start()
        transcription_thread.join()

        return result_queue.get()

    def ask_ollama(self, prompt, responseCallback):
        logging.info(f"Asking OLLaMa with prompt: {prompt}")
        full_prompt = prompt if hasattr(self, "contextSent") else (prompt)
        self.contextSent = True
        jsonParam = {
            "model": self.config.ollama.model,
            "stream": True,
            "context": self.context,
            "prompt": full_prompt
        }
        
        try:
            response = requests.post(self.config.ollama.url,
                                    json=jsonParam,
                                    headers=OLLAMA_REST_HEADERS,
                                    stream=True,
                                    timeout=30)  # Increase the timeout value
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                body = json.loads(line)
                token = body.get('response', '')
                full_response += token

                if 'error' in body:
                    logging.error(f"Error from OLLaMa: {body['error']}")
                    responseCallback("Error: " + body['error'])
                    return

                if body.get('done', False) and 'context' in body:
                    self.context = body['context']
                    break

            responseCallback(full_response.strip())

        except requests.exceptions.ReadTimeout as e:
            logging.error(f"ReadTimeout occurred while asking OLLaMa: {str(e)}")
            responseCallback("Sorry, the request timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            logging.error(f"An error occurred while asking OLLaMa: {str(e)}")
            responseCallback("Sorry, an error occurred. Please try again.")

    def text_to_speech(self, text):
        logging.info(f"Converting text to speech: {text}")
        print('\nAI:\n', text.strip())
        
        # Add AI response to chat history
        self.chat_history.append({
            "timestamp": datetime.now().isoformat(),
            "speaker": "assistant",
            "message": text.strip()
        })

        def play_speech():
            try:
                logging.info("Initializing TTS engine")
                engine = pyttsx3.init()
                
                # Adjust the speech rate (optional)
                rate = engine.getProperty('rate')
                engine.setProperty('rate', rate - 50)  # Decrease the rate by 50 units
                
                # Add a short delay before converting text to speech
                time.sleep(0.5)  # Adjust the delay as needed
                
                logging.info("Converting text to speech")
                engine.say(text)
                engine.runAndWait()
                logging.info("Speech playback completed")
            except Exception as e:
                logging.error(f"An error occurred during speech playback: {str(e)}")

        speech_thread = threading.Thread(target=play_speech)
        speech_thread.start()

    def generate_summary(self, chat_history, summary_type="short"):
        """Generate a summary of the chat history using Ollama"""
        if not chat_history:
            return "Empty conversation"
        
        # Create a conversation text from chat history
        conversation_text = ""
        for entry in chat_history:
            speaker = "User" if entry["speaker"] == "user" else "Assistant"
            conversation_text += f"{speaker}: {entry['message']}\n"
        
        if summary_type == "short":
            prompt = f"""Please provide a very brief summary of this conversation in 3-12 words that could be used as a filename. Use only letters, numbers, and hyphens. Do not use special characters or spaces.

Conversation:
{conversation_text}

Provide only the short summary, nothing else:"""
        else:
            prompt = f"""Please provide a comprehensive summary of this conversation, including the main topics discussed, key points raised, and any conclusions or outcomes. This should be 2-3 sentences.

Conversation:
{conversation_text}

Summary:"""
        
        try:
            jsonParam = {
                "model": self.config.ollama.model,
                "stream": False,
                "prompt": prompt
            }
            
            response = requests.post(self.config.ollama.url,
                                    json=jsonParam,
                                    headers=OLLAMA_REST_HEADERS,
                                    timeout=30)
            response.raise_for_status()
            
            result = response.json()
            summary = result.get('response', '').strip()
            
            if summary_type == "short":
                # Clean up the short summary for filename use
                summary = summary.replace(' ', '-').replace('_', '-')
                # Remove any special characters except hyphens
                import re
                summary = re.sub(r'[^a-zA-Z0-9-]', '', summary)
                summary = summary[:50]  # Limit length
                
            return summary
            
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            if summary_type == "short":
                return "conversation"
            else:
                return "Summary not available due to an error."

    def save_chat_history(self):
        """Save chat history and transcriptions to a file with current datetime as filename"""
        if not self.chat_history and not self.transcriptions:
            return
        
        # Create chat_history directory if it doesn't exist
        chat_dir = "chat_history"
        os.makedirs(chat_dir, exist_ok=True)
        
        # Generate summaries
        logging.info("Generating chat summaries...")
        short_summary = self.generate_summary(self.chat_history, "short")
        long_summary = self.generate_summary(self.chat_history, "long")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(chat_dir, f"{timestamp}_{short_summary}.json")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "summary": long_summary,
                    "timestamp": datetime.now().isoformat(),
                    "chat_history": self.chat_history,
                    "transcriptions": self.transcriptions
                }, f, indent=2, ensure_ascii=False)
            logging.info(f"Chat history and transcriptions saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving chat history: {str(e)}")

def main():
    logging.info("Starting Assistant")
    pygame.init()

    ass = Assistant()

    push_to_talk_with_agent_key = pygame.K_a
    transcribe_only_key = pygame.K_SPACE
    quit_key = pygame.K_ESCAPE

    while True:
        ass.clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == push_to_talk_with_agent_key:
                    logging.info("Push-to-talk-with-agent key pressed")
                    speech = ass.waveform_from_mic(push_to_talk_with_agent_key)

                    transcription = ass.speech_to_text(waveform=speech)

                    ass.ask_ollama(transcription, ass.text_to_speech)

                    time.sleep(1)
                    ass.display_ready_screen()

                elif event.key == transcribe_only_key:
                    logging.info("Transcribe-only key pressed")
                    speech = ass.waveform_from_mic(transcribe_only_key)
                    
                    transcription = ass.transcribe_only(waveform=speech)
                    
                    # Display the transcription on screen
                    ass.display_message(f"Transcribed: {transcription}")
                    
                    time.sleep(3)  # Show transcription for 3 seconds
                    ass.display_ready_screen()

                elif event.key == quit_key:
                    logging.info("Quit key pressed")
                    ass.shutdown()



if __name__ == "__main__":
    main()
