



`setup_speakers` reads from `speakers_samples/`: each **folder** is a speaker (folder name = speaker name). Put any **.mp3** files in that folder; **filenames don't matter** (e.g. `spoken_noise.mp3`, `anything.mp3`). The script enrolls each speaker with **multi-condition** diversity: it embeds every file in the folder, then averages the embeddings.

Example layout:
- `speakers_samples/ivan/spoken_noise.mp3` (and any other .mp3 in `ivan/`)
- `speakers_samples/Alice/studio.mp3`, `speakers_samples/Alice/office.mp3`, …

Diarization: the "enrollment" (training in voices) script is freely available from HuggingFace; the live app loads `speaker_db.pkl` produced by it.


tech
- `pyannote/embedding` converts arbitrary lengthed audio clips of a single person into a vector repsentation#
- - background noise is okay, by influding it AND during quite we can make it more robust against background noise (quiet, loud non-human, loud voices)
- - => you can either concatenate them into one embedding with `enroll_speaker_multi_condition` (otherwise could auto detect conditions and match based on that, but less robust and complex) => (could also do this for moods and stuff)
- - 10-30sec is enough
- - try to do the same device
- - content should match what you are going to do: cover all phonemes for sure 
- - must be the single speaker(we could theoretically both speakers and create a hybrid of both)
- `pyannote/speaker-diarization-3.1` is multiple models stacked
- - segmentation model of speech vs non-speech, general speaker changes and overlaps
- - embedding model of the same kind as used in enrollment
- - clustering algorithm which takes the vectors created from the segmentation model with the ones of the embedding model to label them - automatically labels the not recognized things as "unknown"
- => nooks and cranies
- - sometimes clips are way too short, so it labels the thing as SHORT - embedding models have a minimum length (can be) => (must be interesting ways to increase this)
- `test_speakers.py`should take an audio recording of a conversatiion between some of the speakers + "strangers"

main.py is the actual loop that does stuff - prints to console who is talking and once conversation starts, prints keyword notification
- continiously sends x second chunks into queue
- starts the conversation when the Voice Activity Detection (VAD) model detects speaking
- 3 threads: detects keywords, takes screenshots for "gemini, take a screenshot" or when a verbal cue is detected, notes down todos (by sending chunks of text from the transctipt)
- compiles that after the conversation ends into a summary with the image data
- returns  a string of summary (without todos) and squashes it with todos

how to authenticate with HF? you can either use the login option for a session or pass the token parameter
- go to the pages of 
    - https://huggingface.co/pyannote/embedding
    - https://huggingface.co/pyannote/speaker-diarization
    - https://huggingface.co/pyannote/speaker-diarization-3.1
    - https://huggingface.co/pyannote/speaker-diarization-3.0
    - https://huggingface.co/pyannote/segmentation-3.0 
- fill in the form to accept the conditions. then create a read-key
- pyannote uses old auth functions, so we had to downgrade it


test-sets
- 00.mp3: joey ivan and one stranger


```
=== SECTION 1: Phonetically Balanced (30 seconds) ===
The quick brown fox jumps over the lazy dog. She sells seashells by the seashore.
Peter Piper picked a peck of pickled peppers. How much wood would a woodchuck chuck?
The sixth sick sheik's sixth sheep is sick. Toy boat, toy boat, toy boat.

=== SECTION 2: Varied Intonation (30 seconds) ===
Hello, how are you today? I'm doing great, thanks for asking!
Wait, what did you just say? That's absolutely incredible!
Could you please repeat that? I'm not sure I understand.
Let me think about this for a moment... Yes, that makes perfect sense now.

=== SECTION 3: Conversational Speech (30 seconds) ===
So yesterday I was thinking about grabbing coffee around three or four,
but then my meeting ran late and I completely forgot.
Anyway, the point is we should probably reschedule for next week.
Does Tuesday or Wednesday work better for you? Let me know what you think.

=== SECTION 4: Technical/Domain Specific (30 seconds) ===
The database migration completed successfully at midnight.
We need to update the configuration files before deployment.
Please review the documentation and submit your feedback by Friday.
The system is running optimally with ninety-nine percent uptime.

=== SECTION 5: Numbers and Alphanumeric (20 seconds) ===
My number is five-five-five, two-one-three-four.
The confirmation code is alpha-bravo-seven-charlie-nine.
Please transfer three thousand four hundred fifty-six dollars.
The meeting is scheduled for November fifteenth at two-thirty PM.

=== SECTION 6: Emotional Variation (20 seconds) ===
I'm so excited about this opportunity!
Unfortunately, we'll need to postpone the event.
That's really frustrating, but I understand.
Wow, congratulations on the achievement!





=== SECTION 7: Fricatives and Sibilants (25 seconds) ===
The thief seized the fleece from the freezing breeze.
Sushi chefs choose fresh fish seasonally.
Vicious wolves ventured through the valley very cautiously.
Measure the treasure with pleasure, not pressure.

=== SECTION 8: Plosives and Stops (25 seconds) ===
Pack the black backpack with snacks for the track.
Did David deliver the dividend to theividual?
Take the ticket to the cricket wicket at the thicket.
The big dog dug deep ditches behind the bridge.

=== SECTION 9: Nasal Sounds (25 seconds) ===
Many men manage money in Manhattan minimally.
Nine nimble numbats gnaw on noodles near noon.
Morning meetings meandering through mundane minutes.
Running along the long winding canyon in spring.

=== SECTION 10: Vowel Diversity (30 seconds) ===
Beat, bit, bet, bat, boot, but, bought, boat.
The heat made Pete seek the creek to cool his feet.
Oil the coil before joining the royal convoy.
How now brown cow, allow me to avow my vow.

=== SECTION 11: Connected Speech / Reductions (30 seconds) ===
I'm gonna grab something to eat, wanna come?
Shoulda, woulda, coulda - but we didn't, did we?
Whatcha doing later? D'you wanna meet up?
I'll probably hafta check with 'em before I can say for sure.

=== SECTION 12: Workplace Scenarios (30 seconds) ===
Could you send me that spreadsheet by end of day?
I'll be out of office next Monday through Wednesday.
Let's circle back on this after we sync with the team.
Can you hop on a quick call to discuss the deliverables?

=== SECTION 13: Casual Conversation (30 seconds) ===
Yeah, I totally get what you're saying, but like, honestly?
I mean, it's not that big of a deal, you know?
Oh my god, did you see what happened yesterday?
No way! Are you serious? That's crazy!

=== SECTION 14: Longer Sentences (30 seconds) ===
After considering all the available options and weighing the potential consequences, 
I've decided to proceed with the original plan despite the challenges.
The reality is that we need to fundamentally rethink our approach 
if we want to achieve meaningful results within the timeframe
```



setup
- brew install portaudio