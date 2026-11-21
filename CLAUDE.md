# CamAudioServer

An ESP32 streams video (MJPEG, port 80) and audio (raw 32-bit PCM, port 81). `client.py` consumes both on a computer. It runs VAD and Whisper on the audio and keeps a rolling history of video frames. The long-term goal is to let an LLM answer questions like "What was I holding up when I said that?" by lining up utterances with the frames from the same moment.

## How to work in this repo

- **Make small, incremental changes.** Do one logical step at a time and stop so the user can review it. Don't bundle in the next step, even if it's obvious. If a task mentions things to do "later", leave them alone.
- **Don't touch code outside the task.** No drive-by refactors, renames, or reformatting. If something else needs fixing, mention it instead of changing it.
- **Keep code readable.** Prefer plain, obvious code over clever code. Use descriptive names, named constants instead of magic numbers, and short functions that do one thing. Add a brief comment where the *why* isn't obvious.
- **Keep code organized.** Put constants at the top, then data types, then classes, then functions, with `main()` last. New behavior goes where it logically belongs. Suggest splitting a file when it gets bloated, but don't do it without being asked.
- **Match the existing style.** Follow the conventions already in the code (type hints, lowercase comments, constant naming).
- **Be explicit about threading.** Say which thread owns each piece of shared state. Add locks only when a second thread actually needs access.
- **Explain what you changed.** After a change, say briefly what you did and why, and flag anything you're unsure about, so the reviewer knows where to look.
- **Verify.** Run what you can (e.g. a syntax check). Say clearly if something could not be tested, such as anything needing the live ESP32.

## Commits

Only commit when the user asks. Keep each commit focused on one step.
