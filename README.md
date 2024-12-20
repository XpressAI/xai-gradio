<p align="center">
  <a href="https://github.com/XpressAI/xircuits/tree/master/xai_components#xircuits-component-library-list">Component Libraries</a> •
  <a href="https://github.com/XpressAI/xircuits/tree/master/project-templates#xircuits-project-templates-list">Project Templates</a>
  <br>
  <a href="https://xircuits.io/">Docs</a> •
  <a href="https://xircuits.io/docs/Installation">Install</a> •
  <a href="https://xircuits.io/docs/category/tutorials">Tutorials</a> •
  <a href="https://xircuits.io/docs/category/developer-guide">Developer Guides</a> •
  <a href="https://github.com/XpressAI/xircuits/blob/master/CONTRIBUTING.md">Contribute</a> •
  <a href="https://www.xpress.ai/blog/">Blog</a> •
  <a href="https://discord.com/invite/vgEg2ZtxCw">Discord</a>
</p>






<p align="center"><i>Xircuits Component Library to interface with Gradio! Create interactive user interfaces effortlessly.</i></p>

---
### Xircuits Component Library for Gradio
This library integrates Gradio with Xircuits, enabling the creation of customizable user interfaces, including chatbots, input components, and visual outputs. It supports seamless interaction with machine learning models and real-time data processing.

## Table of Contents

- [Preview](#preview)
- [Prerequisites](#prerequisites)
- [How It Works](#main-xircuits-components)
- [Main Xircuits Components](#main-xircuits-components)
- [Try the Examples](#try-the-examples)
- [Installation](#installation)

## Preview

### Cat Adoption Form Example

![CatAdoptionForm](https://github.com/user-attachments/assets/f33e3507-7bad-4fe3-aa40-9f9eff647c86)

### CatAdoptionForm Result

![CatAdoptionForm_result](https://github.com/user-attachments/assets/f6d9d7e5-119d-49ea-8ee4-fbb1bd17d8f3)

### Data Collection Interface Example

![DataCollectionInterface](https://github.com/user-attachments/assets/ba0d1163-13db-4f5e-aa30-c666c1b9b132)

### Data Collection Interface Result

![DataCollectionInterface_result](https://github.com/user-attachments/assets/d686efab-9f87-4e61-a5b5-02ab055589f7)

### Gradio Chat Example

![GradioChat](https://github.com/user-attachments/assets/582faa00-f529-4618-b06d-8ad42d7f996c)

### Gradio Chat Result

![GradioChat_result](https://github.com/user-attachments/assets/4a7a4e09-130a-4d9b-803c-f80ffc407f78)

### Gradio Agent Example

![GradioAgentChat](https://github.com/user-attachments/assets/0028e06e-a4be-4e21-a480-f6a96dec5771)

![Tools](https://github.com/user-attachments/assets/3a27a817-d110-4ab6-be91-f0b0beda710d)

### Gradio Agent Result

![GradioAgentChat_example](https://github.com/user-attachments/assets/76d715af-5d37-4ba3-9545-37c883b85d98)

## Prerequisites

Before you begin, you will need the following:

1. Python3.9+.
2. Xircuits.


### How it Works
The Gradio component library follows this workflow pattern:

1. **Input Components**: 
Select input components from the component library (e.g., `GradioTextbox`, `GradioImage`, `GradioSlider`, etc.) based on your interface needs.

2. **Output Components**:
Choose output components (e.g., `GradioTextbox`, `GradioJSONOutput`, `GradioPlotOutput`, etc.) to display your results.

3. **GradioInterface Connection**: 
- Connect input components to `GradioInterface`'s `inputs` InPort
- Connect output components to `GradioInterface`'s `outputs` InPort
- Define corresponding parameter names in `parameterNames` InPort

4. **Completing the Flow**:
- Connect `GradioInterface`'s interface outPort to `GradioLaunch`
- Connect processing logic to `GradioInterface`'s `fn` branch
- Use `GradioFnReturn` to process and return values

## Main Xircuits Components

### GradioChatInterface Component:
Creates a chatbot interface supporting text and image uploads.

<img src="https://github.com/user-attachments/assets/36e38aec-42df-4585-9db1-85e45dae1969" alt="GradioChatInterface" width="200" height="150" />

### GradioFnReturn Component:
Processes return values for a Gradio function, integrates with OpenAI or custom agent responses, and appends them to chat history.

<img src="https://github.com/user-attachments/assets/6431c783-c1b0-4369-a481-67fe96da6217" alt="GradioFnReturn" width="200" height="200" />

### GradioInterface Component:
Builds a Gradio interface using dynamic inputs and outputs, allowing flexible integration with workflows.

### GradioLaunch Component:
Launches a Gradio interface or block as a standalone app.

## Try The Examples

### Cat Adoption Form Example  
 This example creates a cat adoption form using Gradio components. Users input the cat's name, adoption date, age, vaccination status, and upload an image. Submitted details are displayed alongside the image, and data can be saved by clicking the "Save" button.

### Data Collection Interface Example

This example creates an interactive form using Gradio components to collect user data, including name, gender, age, language, and voice recording. Submitted data is displayed as a result, and the workflow supports saving the inputs for further use.

### Gradio Chat Example

This example creates an interactive Gradio chatbot interface that integrates OpenAI responses and predefined replies. The bot can respond to common greetings such as "hello" or "bye" using predefined responses and utilizes OpenAI's GPT for more complex queries. The chat interface is interactive, tracks conversation history, and allows seamless user interaction.

### Gradio Agent Example  

This example creates an interactive Gradio chatbot interface powered by an OpenAI agent. The bot can respond to user queries, such as providing current weather information and the time in specific locations, by using predefined tools like `get_weather` and `get_current_time`. The chatbot tracks the conversation history and saves the responses for further interaction.

## Installation
To use this component library, ensure that you have an existing [Xircuits setup](https://xircuits.io/docs/main/Installation). You can then install the Gradio library using the [component library interface](https://xircuits.io/docs/component-library/installation#installation-using-the-xircuits-library-interface), or through the CLI using:

```
xircuits install gradio
```
You can also do it manually by cloning and installing it:
```
# base Xircuits directory
git clone https://github.com/XpressAI/xai-gradio xai_components/xai_gradio 
pip install -r xai_components/xai_gradio/requirements.txt 
```
