
First time submodule init:  
  `git submodule add https://github.com/mlc-ai/tokenizers-cpp "external/tokenizers-cpp"`  


Build:  
`cmake -S . --preset=release`  
`cmake --build --preset=release`  

Run:  
 `build\release\bin\chatterbox_tts_demo.exe -m models\onnx --download -t "Hello, my friend! If you seek knowledge about mighty Talos, you have most certainly come to the right person." -v assets\malebrute.wav -o test_q4.wav --dtype q4`  


Cache conditionals from wav files found in "assets" folder to disk ("cache" folder):  
`build\release\bin\chatterbox_tts_demo.exe --precache`

Add some tags:  
 `build\release\bin\chatterbox_tts_demo.exe -m models\onnx -t"[clear throat] Listen up worm! [sniff] ugh! Did you just [groan] soil yourself? [cough] Well, [sigh] I was just going to disembowel you anyway. So, like, [laugh] whatever!"  -v babette
 -o test_q4.wav --dtype q4`  

Notes:
- ONNX runtime threads are being set to CPU cores / 4  with a minimum of 2
- currently CPU only (due to current ONNX libs) but still managing 0.75 RTF!!! (at least on my system)
- other quant types are available (fp32,q8,q4f16,q4) 
  - q4 seems best on CPU (can test using the --dtype flag)
- ChatterBox Turbo is currently English only (multilingual is coming supposedly)
- saving/caching of cloned conditionals has not been done (TODO item)
  - TODO items in PROJECT_TTS_ADDON.md
- known tags:
  - `[clear throat]`, `[sigh]`, `[shush]`, `[cough]`, `[groan]`, `[sniff]`, `[gasp]` ,`[chuckle]`, `[laugh]`
  
Some Updates:  
- onnxruntime higher than v1.22.0 needed to support some features
  - using v1.23.2 (actually 1.24.0 pre)
 ---
Sources:  
ChatterBox:
 - https://github.com/resemble-ai/chatterbox
 - https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX/tree/main

tokenizers-cpp:
- https://github.com/mlc-ai/tokenizers-cpp
- Overwite the submodules needed by tokenizers-cpp with current versions from vcpkg:
  - SentencePiece:
    - https://github.com/google/sentencepiece
  - msgpack for C++ (cpp_master branch)
    - https://github.com/msgpack/msgpack-c/tree/cpp_master

