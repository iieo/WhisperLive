from whisper_live.client import TranscriptionClient

# Client initialisieren
client = TranscriptionClient(
    "https://stt.iieo.de",  # Server-Adresse
    80,         # Server-Port
    lang="de",    # Sprache (nur bei multilingualen Modellen)
    translate=False,  # Auf True setzen für Übersetzung ins Englische
    model="small",    # Whisper-Modellgröße
    use_vad=False,    # Voice Activity Detection
    save_output_recording=True,  # Mikrofon-Input als .wav speichern
    output_recording_filename="./output_recording.wav",
    mute_audio_playback=False,   # Audio stumm schalten bei Dateien
    enable_translation=True,     # Übersetzung aktivieren
    target_language="de",        # Zielsprache für Übersetzung
)
client()
