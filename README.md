# HuoziIME: An On-Device LLM-Enhanced Input Method for Deep Personalization

<div align="center" style="line-height: 1;">
    <a href="#source-code--installation" style="margin: 2px;">
        <img alt="WeChat" src="https://img.shields.io/badge/WeChat-Coming_Soon-%2307C160?logo=wechat&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://github.com/Shan-HIT/HuoziIME" target="_blank" style="margin: 2px;">
        <img alt="GitHub" src="https://img.shields.io/badge/HuoziIME-GitHub-black?logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="LICENSE" style="margin: 2px;">
        <img alt="Code License" src="https://img.shields.io/badge/Code%20License-GPLv3-blue.svg" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://github.com/Shan-HIT/HuoziIME/stargazers" target="_blank" style="margin: 2px;">
        <img alt="Stars" src="https://img.shields.io/github/stars/Shan-HIT/HuoziIME?style=flat&logo=github" style="display: inline-block; vertical-align: middle;"/>
    </a>
</div>

## Demo Preview

<div align="center">
<video src="https://github.com/user-attachments/assets/6e8e1270-d859-4d7b-ac4f-5266201af3e6" controls="controls" width="100%">
  </video>
</div>

## Build Environment

| Component | Version |
| :--- | :--- |
| Android SDK | `36.1.0`, `35.0.1`, `35.0.0`, `34.0.0` |
| Android NDK | `25.2.9519653` |
| CMake | `3.22.1` |

This project is a secondary development based on [YuyanIme](https://github.com/gurecn/YuyanIme), and it is distributed under the **GPL v3.0** license.

## Build from Source

### Option A: Android Studio (Recommended)

1. Clone the repository:

   ```bash
   git clone https://github.com/Shan-HIT/HuoziIME.git
   cd HuoziIME
   ```

2. Open the project in Android Studio.
3. Make sure the required SDK, NDK, and CMake versions above are installed.
4. In `Build Variants`, choose one variant:
   - `onlineDebug` / `onlineRelease`
   - `offlineDebug` / `offlineRelease`
5. Build and run on your Android device.

### Option B: Command Line (If local `gradle` is configured)

```bash
gradle :app:assembleOfflineDebug
gradle :app:assembleOnlineDebug
```

## Source Code & Installation

| Content | Status | Note |
| :--- | :---: | :--- |
| **Source Code** | ![Open Source](https://img.shields.io/badge/Status-Open%20Source-success) | The core codebase is already public in this repository. |
| **APK Installer** | ![Released](https://img.shields.io/badge/Status-Released-success) | v1.0.0-beta is now available in Releases. |
---
## Acknowledgements

This project stands on the shoulders of several excellent open source projects. Their ideas, code, and tooling made the development of HuoziIME possible.

<div align="center">

| Project | Contribution |
| :--- | :--- |
| [**YuyanIME**](https://github.com/gurecn/YuyanIme) | The original input method foundation that inspired and enabled the secondary development of HuoziIME. |
| [**YuyanSDK**](https://github.com/gurecn/yuyansdk) | Core SDK support used for building the intelligent input experience and related capabilities. |
| [**llama.cpp**](https://github.com/ggml-org/llama.cpp) | Efficient on-device LLM inference support that helps bring local model capability to the IME workflow. |
| [**Qwen**](https://github.com/QwenLM/Qwen3) | The open model family that provides the language intelligence foundation behind the LLM-enhanced experience. |

</div>

> We sincerely thank the maintainers and contributors of these projects for advancing open source research and engineering.
---
### Stay Tuned

If you are interested in this project, please click **Star** or **Watch** on GitHub to receive the latest updates.
