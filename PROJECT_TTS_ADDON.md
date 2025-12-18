## Todo:

Performance:
- current RTF is ~0.75; is 0.5 possible?
- can any functions be done in parallel?

VoiceConditionals caching:
- LRU in-memory cache.
- on in-memory cache miss check VoiceConditionals disk cache
  - on disk cache miss fall back to existing methods of getting audio files
- save new VoiceConditionals to disk 

Models and model files:
- ability to load all needed models and files (tokenizer.json) at once to memory
- ability to unload model and model files on request
- ability to switch models on request


---
### After Integration:
Issue Set 1 to investigate later:
- use existing threadpool for asynchronous saving of new VoiceConditionals to disk (ignore failure just don't leave bad files)
- check if any functions can be done async

Issues Set 2 to investigate later:
- (UI/Config) enable/load models
- (UI/Config) disable/unload models
- (UI/Config) view/change models

Issues Set 3 to investigate later:
- clear VoiceConditionals cache
- remove single entry in VoiceConditionals cache
- (UI) view/display cache resource utilization
- (UI) view/display voice file name in cache