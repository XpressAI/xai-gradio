from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent,secret, xai_component, dynalist, SubGraphExecutor

import gradio as gr
import makefun
from openai import OpenAI
import datetime


@xai_component(color='blue')
class GradioChatInterface(Component):
    """Creates a Chatbot interface with support for text and image uploads."""    
    interface: OutArg[gr.Blocks]
    fn: BaseComponent
    message: OutArg[str]
    history: OutArg[list]
    uploaded_image: OutArg[any] 

    def execute(self, ctx) -> None:
        

        def fn(history=None, message=None, image=None):
            if history is None:
                history = []

            if image is not None:
                self.uploaded_image.value = image
                history.append((None, gr.Image(value=image, label="User's image")))

            if message:
                self.message.value = message
                history.append((message, None))
                SubGraphExecutor(self.fn).do(ctx)
                result = ctx['__gradio__result__']

                history.append((None, result[0]))

            return history, "", None

        with gr.Blocks() as demo:
            chatbot = gr.Chatbot(label="Chatbot")
            msg = gr.Textbox(label="Type a message", placeholder="Enter your message here...", interactive=True, lines=1)
            img = gr.Image(label="Upload an image", type="filepath", interactive=True)
            submit = gr.Button("Send")

            submit.click(fn, inputs=[chatbot, msg, img], outputs=[chatbot, msg, img])

            msg.submit(fn, inputs=[chatbot, msg, img], outputs=[chatbot, msg, img])

        self.interface.value = demo

@xai_component(color='blue')
class GradioInterface(Component):
    parameterNames: InArg[dynalist]
    inputs: InArg[dynalist]
    outputs: InArg[dynalist]
    interface: OutArg[gr.Blocks]
    fn: BaseComponent
    parameters: OutArg[dict]

    def execute(self, ctx) -> None:
        assert len(self.parameterNames.value) == len(self.inputs.value), "Parameter names must match the number of inputs"

        def process_inputs(*args, **kwargs):
            input_data = kwargs
            outputs = []
            if len(self.outputs.value) == 1:
                outputs.append(input_data)
            elif len(self.outputs.value) > 1:
                first_output_data = {k: v for k, v in input_data.items() if k != "Image"}
                outputs.append(first_output_data)
                second_output_data = input_data.get("Image", None)
                outputs.append(second_output_data)
            else:
                raise ValueError("No output components are defined!")
            return outputs

        def fn_impl(*args, **kwargs):
            mapped_outputs = process_inputs(*args, **kwargs)
            self.parameters.value = kwargs
            SubGraphExecutor(self.fn).do(ctx)
            result = ctx.get('__gradio__result__', [{}] * len(self.outputs.value))
            final_outputs = []
            for i in range(len(self.outputs.value)):
                final_outputs.append(mapped_outputs[i] if i < len(mapped_outputs) else result[i])
            return tuple(final_outputs)

        sig = "fn(%s)" % (",".join(self.parameterNames.value))
        fn = makefun.create_function(sig, fn_impl)
        self.interface.value = gr.Interface(fn=fn, inputs=self.inputs.value, outputs=self.outputs.value)
    
        self.interface.value = gr.Interface(
            fn=fn,
            inputs=self.inputs.value,
            outputs=self.outputs.value,
            allow_flagging="manual",  
            flagging_options=[("Save", "save_value")]   
        )


@xai_component(color='blue')
class GradioLaunch(Component):
    """ Launches a Gradio Interface or Block

    ##### inPorts:
    - app (Interface | Block): interface or block to launch
    """
    app: InArg[gr.Blocks]

    def execute(self, ctx) -> None:
        self.app.value.queue().launch()

@xai_component(color='purple')
class GradioFnReturn(Component):
    """Collects return values for a Gradio function, integrates predefined responses from ctx,
    OpenAI, Agent, and appends chat history with timestamps.

    ##### inPorts:
    - results (list[any]): Results for the Gradio function.
    - message (str): The user message to process.
    - history (list): Chat history to update and save (optional).
    - use_responses (bool): Whether to use predefined responses from ctx.
    - use_openai (bool): Whether to use OpenAI for generating responses.
    - use_agent (bool): Whether to use Agent for generating responses.
    - log_file (str): Path to the file where the conversation history will be saved (optional).
    """

    results: InArg[dynalist]
    message: InArg[str]
    history: InArg[list]
    use_responses: InArg[bool]  # Enable or disable predefined responses
    use_openai: InArg[bool]  # Enable or disable OpenAI integration
    use_agent: InArg[bool]  # Enable or disable Agent integration
    log_file: InArg[str]  # Path for saving the log file

    def execute(self, ctx) -> None:
        import datetime

        # Inputs
        message = self.message.value.strip() if self.message.value else None
        chat_history = self.history.value or []
        use_responses = self.use_responses.value
        use_openai = self.use_openai.value
        use_agent = self.use_agent.value
        log_file_path = self.log_file.value or "chat_history.log"

        reply = None

        # Step 1: Agent Integration
        if use_agent and message:
            try:
                agent_context = ctx.get('agent', None)
                agent_run = ctx.get('agent_run', None)

                if not agent_context or not agent_run:
                    raise ValueError("AgentRun or agent context not found in ctx. Please initialize them first.")

                conversation = ctx.get('agent_conversation', [])
                conversation.append({"role": "user", "content": message})
                agent_run.conversation.value = conversation
                agent_run.agent_name.value = agent_context['name']

                agent_run.execute(ctx)
                reply = agent_run.last_response.value
                ctx['agent_conversation'] = agent_run.out_conversation.value

            except Exception as e:
                print(f"Agent Error: {str(e)}")

        if  not reply and use_openai:
            try:
                client = ctx.get('client', None)
                if not client:
                    print("OpenAI client is not configured in the context. Skipping OpenAI integration.")
                else:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": message}
                        ],
                        max_tokens=150,
                    )
                    reply = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI Error: {str(e)}")

        # Step 3: Predefined Responses
        if not reply and message:
            responses_dict = ctx.get('gradio_responses', {})
            if responses_dict:
                for keyword, response in responses_dict.items():
                    if keyword in message.lower():
                        reply = response
                        break
                if not reply:
                    reply = "I don't understand."

        # Step 4: Default Fallback
        if not reply:
            reply = self.results.value[0] if self.results.value else "No result found"

        # Update Chat History
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry_user = f"[{timestamp}] User: {message}" if message else None
        new_entry_bot = f"[{timestamp}] Bot: {reply}"
        updated_history = chat_history + [new_entry_user, new_entry_bot] if message else chat_history

        # Save Result to Context
        ctx['__gradio__result__'] = [reply]
        ctx['history'] = updated_history

        # Save Chat Log
        if message:
            try:
                with open(log_file_path, "a") as file:
                    if new_entry_user:
                        file.write(f"{new_entry_user}\n")
                    file.write(f"{new_entry_bot}\n")
                print(f"Chat history saved to {log_file_path}.")
            except Exception as e:
                print(f"Failed to write log to {log_file_path}: {str(e)}")


@xai_component(color='orange')
class GradioTextbox(Component):
    """ Creates a Textbox """
    label: InCompArg[str]
    component: OutArg[any]
    
    def execute(self, ctx) -> None:
        self.component.value = gr.Textbox(label=self.label.value)

@xai_component(color='orange')
class GradioRadioButton(Component):
    """ Creates a Radio Button with several choices """
    choices: InArg[dynalist]
    component: OutArg[any]
    
    def execute(self, ctx) -> None:
        self.component.value = gr.Radio(choices=self.choices.value)

@xai_component(color='orange')
class GradioSlider(Component):
    """Creates a Gradio Slider."""
    label: InCompArg[str]
    min_value: InArg[int]
    max_value: InArg[int]
    step: InArg[int]
    component: OutArg[any]

    def execute(self, ctx) -> None:

        self.component.value = gr.Slider(
            minimum=self.min_value.value or 0,
            maximum=self.max_value.value or 100,
            step=self.step.value or 1,
            label=self.label.value or "Slider"
        )

@xai_component(color='orange')
class GradioDropdown(Component):
    """Creates a Gradio Dropdown."""
    label: InCompArg[str]
    choices: InArg[dynalist]
    component: OutArg[any]

    def execute(self, ctx) -> None:

        self.component.value = gr.Dropdown(
            choices=self.choices.value or [],
            label=self.label.value or "Dropdown"
        )

@xai_component(color='orange')
class GradioCheckbox(Component):
    """Creates a Gradio Checkbox."""
    label: InCompArg[str]
    value: InArg[bool]
    component: OutArg[any]

    def execute(self, ctx) -> None:
        self.component.value = gr.Checkbox(
            value=self.value.value or False,
            label=self.label.value or "Checkbox"
        )

@xai_component(color='orange')
class GradioCheckboxGroup(Component):
    """Creates a Gradio Checkbox Group."""
    label: InCompArg[str]
    choices: InArg[dynalist]
    component: OutArg[any]

    def execute(self, ctx) -> None:
        self.component.value = gr.CheckboxGroup(
            choices=self.choices.value or [],
            label=self.label.value or "Checkbox Group"
        )

@xai_component(color='orange')
class GradioNumber(Component):
    """Creates a Gradio Number Input."""
    label: InArg[str]
    value: InArg[float]
    allow_negative: InArg[bool]  
    component: OutArg[any]

    def execute(self, ctx) -> None:
        if self.allow_negative.value:
            self.component.value = gr.Number(
                value=self.value.value or 0,
                label=self.label.value or "Number Input"
            )
        else:
            self.component.value = gr.Number(
                value=self.value.value or 0,
                label=self.label.value or "Number Input",
                minimum=0 
            )

@xai_component(color='orange')
class GradioFile(Component):
    """Creates a Gradio File Upload."""
    label: InCompArg[str]
    file_types: InArg[dynalist]
    component: OutArg[any]

    def execute(self, ctx) -> None:
        self.component.value = gr.File(
            file_types=self.file_types.value or None,
            label=self.label.value or "Upload File"
        )

@xai_component(color='orange')
class GradioImage(Component):
    """Creates a Gradio Image Input."""
    label: InArg[str]
    type: InArg[str] 
    width: InArg[int]   
    height: InArg[int]  
    component: OutArg[any]

    def execute(self, ctx) -> None:
        self.component.value = gr.Image(
            type=self.type.value or "filepath",
            label=self.label.value or "Image Input",
            width=self.width.value or 200,  
            height=self.height.value or 200  
        )

@xai_component(color='orange')
class GradioAudio(Component):
    """Creates a Gradio Audio Input."""
    label: InCompArg[str]
    type: InArg[str]  # filepath or numpy
    component: OutArg[any]

    def execute(self, ctx) -> None:
        self.component.value = gr.Audio(
            type=self.type.value or "filepath",
            label=self.label.value or "Audio Input"
        )

@xai_component(color='orange')
class GradioVideo(Component):
    """Creates a Gradio Video Input."""
    label: InCompArg[str]
    component: OutArg[any]

    def execute(self, ctx) -> None:
        self.component.value = gr.Video(
            label=self.label.value or "Video Input"
        )

@xai_component(color='orange')
class GradioDateTime(Component):
    """Creates a Gradio Date Picker."""
    label: InArg[str]
    include_time: InArg[bool]
    component: OutArg[any]
    
    def execute(self, ctx) -> None:
        if self.include_time.value:
            self.component.value = gr.DateTime(
                label=self.label.value or "Date and Time Picker"
            )
        else:
            self.component.value = gr.DateTime(
                label=self.label.value or "Date Picker",
                include_time=False
            )

@xai_component(color='orange')
class GradioColorPicker(Component):
    """Creates a Gradio Color Picker."""
    label: InCompArg[str]
    component: OutArg[any]

    def execute(self, ctx) -> None:
        self.component.value = gr.ColorPicker(
            label=self.label.value or "Color Picker"
        )

@xai_component(color='orange')
class GradioAudioRecord(Component):
    """Creates a Gradio Audio Input."""
    label: InCompArg[str]  
    sources: InArg[list] 
    type: InArg[str] 
    component: OutArg[any]  

    def execute(self, ctx) -> None:
        
        self.component.value = gr.Audio(
            label=self.label.value or "Upload or Record Audio",
            sources=self.sources.value or ["microphone", "upload"],
            type=self.type.value or "filepath"
        )

@xai_component(color='orange')
class GradioDataframe(Component):
    """ Creates a Dataframe Input/Output """
    headers: InArg[dynalist]
    datatype: InArg[dynalist]
    component: OutArg[any]

    def execute(self, ctx) -> None:
        self.component.value = gr.Dataframe(
            headers=self.headers.value,
            datatype=self.datatype.value
        )

@xai_component(color='orange')
class GradioJSONOutput(Component):
    """ Creates a JSON Output """
    label: InCompArg[str]
    component: OutArg[any]

    def execute(self, ctx) -> None:
        self.component.value = gr.JSON(label=self.label.value)

@xai_component(color='orange')
class GradioPlotOutput(Component):
    """ Creates a Plot Output """
    label: InCompArg[str]
    component: OutArg[any]

    def execute(self, ctx) -> None:
        self.component.value = gr.Plot(label=self.label.value)

@xai_component(color='orange')
class GradioGallery(Component):
    """ Creates a Gallery Input/Output """
    label: InCompArg[str]
    component: OutArg[any]

    def execute(self, ctx) -> None:
        self.component.value = gr.Gallery(label=self.label.value)

@xai_component(color='purple')
class GradioResponses(Component):
    """Stores a dictionary of predefined responses in the context."""

    responses_dict: InArg[dict]

    def execute(self, ctx) -> None:
        ctx['gradio_responses'] = self.responses_dict.value or {}

