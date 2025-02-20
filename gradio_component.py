from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, secret, xai_component, dynalist, SubGraphExecutor
import gradio as gr
import makefun
import datetime

@xai_component(color='blue', type='branch')
class GradioChatInterface(Component):
    """
    Creates a Gradio Chatbot interface.

    ##### outPorts:
    - interface (gr.Blocks): Gradio Blocks object for the chatbot interface. Connect to `GradioLaunch`.
    - message (str): the most recent user message.
    - history (list): Chat history for the session in OpenAi conversation format.

    ##### Branches:
    - fn (BaseComponent): Subgraph executor for processing messages.
    """
    interface: OutArg[gr.Blocks]
    fn: BaseComponent
    message: OutArg[str]
    history: OutArg[list]

    def execute(self, ctx) -> None:
        def role_dicts_to_pairs(history):
            """
            Converts a list of role/content dicts like:
            [
              {"role": "user", "content": "Hello"},
              {"role": "assistant", "content": "Hi there!"},
              {"role": "user", "content": "How are you?"}
            ]
            into a list of [user_msg, assistant_msg] pairs for Gradio display:
            [
              ["Hello", "Hi there!"],
              ["How are you?", None]
            ]
            """
            display = []
            user_buffer = None
            for item in history:
                role = item.get("role", "")
                content = item.get("content", "")
                if role == "system":
                    continue
                elif role == "user":
                    # If there's a user_buffer pending, append it
                    if user_buffer is not None:
                        display.append([user_buffer, None])
                    user_buffer = content
                elif role == "assistant":
                    if user_buffer is not None:
                        display.append([user_buffer, content])
                        user_buffer = None
                    else:
                        display.append([None, content])
            # Edge case: if user message is leftover
            if user_buffer is not None:
                display.append([user_buffer, None])
            return display

        def submit_fn(user_message=None, user_image=None):
            # Pass the user message to the outports
            if user_message:
                self.message.value = user_message

            # Execute the subgraph
            SubGraphExecutor(self.fn).do(ctx)

            # Now retrieve the updated conversation from ctx
            conversation = ctx.get('__gradio_chat_history__', [])
            self.history.value = conversation  # Store it in the outport

            # Convert conversation into display pairs
            display_history = role_dicts_to_pairs(conversation)

            # Return the display_history to the chatbot, and clear text/image input
            return display_history, "", None

        # Build the Gradio interface
        with gr.Blocks() as demo:
            chatbot = gr.Chatbot(label="Chatbot")
            msg = gr.Textbox(
                label="Type a message", 
                placeholder="Enter your message here...",
                interactive=True, lines=1
            )
            submit = gr.Button("Send")

            submit.click(submit_fn, inputs=[msg], outputs=[chatbot, msg])
            msg.submit(submit_fn, inputs=[msg], outputs=[chatbot, msg])

        self.interface.value = demo

@xai_component(color='blue', type='branch')
class GradioInterface(Component):
    """Creates a custom Gradio Interface.

    ##### inPorts:
    - parameterNames (dynalist): List of parameter names for the function.
    - inputs (dynalist): List of input components.
    - outputs (dynalist): List of output components.

    ##### outPorts:
    - interface (gr.Blocks): The Gradio Interface object.
    - parameters (dict): Dictionary of input parameters. This should be connected to a `GradioFnReturn`'s `results` inPort. 

    ##### Branches:
    - fn (BaseComponent): Subgraph executor that defines the main processing logic for the interface. 
                          Typically, this should be connected to a `GradioFnReturn` component or a similar processing component.
    """
    parameterNames: InArg[dynalist]
    inputs: InArg[dynalist]
    outputs: InArg[dynalist]
    interface: OutArg[gr.Blocks]
    fn: BaseComponent
    parameters: OutArg[dict]

    def execute(self, ctx) -> None:
        assert len(self.parameterNames.value) == len(self.inputs.value), "Parameter names must match the number of inputs"

        def process_inputs(*args, **kwargs):
            """Maps input arguments to expected outputs."""
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
            """Core implementation of the interface function."""
            mapped_outputs = process_inputs(*args, **kwargs)
            self.parameters.value = kwargs
            SubGraphExecutor(self.fn).do(ctx)  # Executes the connected subgraph or component.
            result = ctx.get('__gradio__result__', [{}] * len(self.outputs.value))
            final_outputs = []
            for i in range(len(self.outputs.value)):
                final_outputs.append(mapped_outputs[i] if i < len(mapped_outputs) else result[i])
            return tuple(final_outputs)

        # Dynamically create the function signature based on parameter names
        sig = "fn(%s)" % (",".join(self.parameterNames.value))
        fn = makefun.create_function(sig, fn_impl)

        # Create Gradio Interface
        self.interface.value = gr.Interface(
            fn=fn,
            inputs=self.inputs.value,
            outputs=self.outputs.value,
            allow_flagging="manual",
            flagging_options=[("Save", "save_value")]
        )

@xai_component(color='blue')
class GradioLaunch(Component):
    """Launches a Gradio Interface or Block.

    ##### inPorts:
    - app (Interface | Block): Gradio Interface or Block to launch.
    """
    app: InArg[gr.Blocks]

    def execute(self, ctx) -> None:
        self.app.value.queue().launch()

@xai_component(color='purple')
class GradioFnReturn(Component):
    """
    Collect the return values for a Gradio function.

    ##### inPorts:
    - results (list[any]): Results for the Gradio function.

    The results here get added to the context as `__gradio__result__`.
    """
    results: InArg[dynalist]

    def execute(self, ctx) -> None:
        ctx['__gradio__result__'] = self.results.value

@xai_component(color='purple')
class GradioPredefinedResponses(Component):
    """
    Looks for keywords in the user's message and returns a matching predefined response.
    If no match, returns a fallback text (e.g., "I don't understand.").

    ##### inPorts:
    - message (str): The user's message.
    - responses_dict (dict): A dictionary of {keyword: response}.
    - fallback (str): Text to return if no keyword matches.

    ##### outPorts:
    - reply (str): The response determined by the keywords or fallback.
    """
    message: InArg[str]
    responses_dict: InArg[dict]
    fallback: InArg[str]
    reply: OutArg[str]

    def execute(self, ctx) -> None:
        msg = self.message.value
        responses = self.responses_dict.value or {}
        fallback_text = self.fallback.value or "I don't understand."

        if not msg:
            # If there's no message at all, raise or just set fallback
            # For now, let's set fallback
            self.reply.value = fallback_text
            return

        lower_msg = msg.lower()
        for keyword, response_text in responses.items():
            if keyword in lower_msg:
                self.reply.value = response_text
                return

        # No match found
        self.reply.value = fallback_text

@xai_component(color='purple')
class GradioChatFnReturn(Component):
    """
    Handles the chat history and appends responses from a OpenAI-like Chat to it.
    Internally updates `ctx['__gradio_chat_history__']` for `GradioChatInterface` to process.

    ##### inPorts:
    - message (str): The user's message just sent.
    - history (list): The conversation so far. Typically connected from the `GradioChatInterface` component.
    - llm_response (str/dict): The raw LLM response from the model.
    - log_file (str): Path to optionally log the conversation.

    ##### outPorts:
    """
    message: InArg[str]
    history: InArg[list]
    llm_response: InArg[any]
    log_file: InArg[str]

    def execute(self, ctx) -> None:

        if '__gradio_chat_history__' not in ctx:
            ctx['__gradio_chat_history__'] = []
        conversation = ctx['__gradio_chat_history__']

        user_msg = self.message.value
        if user_msg and user_msg.strip():
            conversation.append({"role": "user", "content": user_msg})

        raw = self.llm_response.value
        if isinstance(raw, dict):
            raw = raw.get("content", "")
        assistant_reply = str(raw).strip()
        if assistant_reply:
            conversation.append({"role": "assistant", "content": assistant_reply})

        ctx['__gradio_chat_history__'] = conversation

        log_file = self.log_file.value or "chat_history.log"
        try:
            with open(log_file, "a") as f:
                f.write(f"User: {user_msg}\nBot: {assistant_reply}\n")
            print(f"LLM conversation logged to {log_file}")
        except Exception as e:
            print(f"Failed to log conversation: {e}")


@xai_component(color='orange')
class GradioTextbox(Component):
    """Creates a Gradio Textbox.

    ##### inPorts:
    - label (str): The label for the Textbox.

    ##### outPorts:
    - component (gr.Textbox): The created Textbox component.
    """
    label: InCompArg[str]
    component: OutArg[any]

    def execute(self, ctx) -> None:
        self.component.value = gr.Textbox(label=self.label.value)

@xai_component(color='orange')
class GradioRadioButton(Component):
    """Creates a Gradio Radio Button.

    ##### inPorts:
    - choices (dynalist): List of choices for the radio button.

    ##### outPorts:
    - component (gr.Radio): The created Radio component.
    """
    choices: InArg[dynalist]
    component: OutArg[any]

    def execute(self, ctx) -> None:
        self.component.value = gr.Radio(choices=self.choices.value)

@xai_component(color='orange')
class GradioSlider(Component):
    """Creates a Gradio Slider.

    ##### inPorts:
    - label (str): The label for the slider.
    - min_value (int): The minimum value of the slider.
    - max_value (int): The maximum value of the slider.
    - step (int): The step size for the slider.

    ##### outPorts:
    - component (gr.Slider): The created Slider component.
    """
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
    """Creates a Gradio Dropdown.

    ##### inPorts:
    - label (str): The label for the dropdown.
    - choices (dynalist): List of options for the dropdown.

    ##### outPorts:
    - component (gr.Dropdown): The created Dropdown component.
    """
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
    """Creates a Gradio Checkbox.

    ##### inPorts:
    - label (str): The label for the checkbox.
    - value (bool): Default state of the checkbox (checked or unchecked).

    ##### outPorts:
    - component (gr.Checkbox): The created Checkbox component.
    """
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
    """Creates a Gradio Checkbox Group.

    ##### inPorts:
    - label (str): The label for the group.
    - choices (dynalist): List of options in the group.

    ##### outPorts:
    - component (gr.CheckboxGroup): The created CheckboxGroup component.
    """
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
    """Creates a Gradio Number Input.

    ##### inPorts:
    - label (str): The label for the input.
    - value (float): Default value of the input.
    - allow_negative (bool): Whether to allow negative numbers.

    ##### outPorts:
    - component (gr.Number): The created Number component.
    """
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
    """Creates a Gradio File Upload Input.

    ##### inPorts:
    - label (str): The label for the upload field.
    - file_types (dynalist): List of allowed file types.

    ##### outPorts:
    - component (gr.File): The created File component.
    """
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
    """Creates an Image Input.

    ##### inPorts:
    - label (str): The label for the image input.
    - type (str): The type of input ("filepath" or "numpy").
    - width (int): Width of the input (optional).
    - height (int): Height of the input (optional).

    ##### outPorts:
    - component (gr.Image): The created Image component.
    """
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
    """Creates an Audio Input.

    ##### inPorts:
    - label (str): The label for the audio input.
    - type (str): Type of input ("filepath" or "numpy").

    ##### outPorts:
    - component (gr.Audio): The created Audio component.
    """
    label: InCompArg[str]
    type: InArg[str]
    component: OutArg[any]

    def execute(self, ctx) -> None:
        self.component.value = gr.Audio(
            type=self.type.value or "filepath",
            label=self.label.value or "Audio Input"
        )

@xai_component(color='orange')
class GradioVideo(Component):
    """Creates a Gradio Video Input.

    ##### inPorts:
    - label (str): The label for the video input.

    ##### outPorts:
    - component (gr.Video): The created Video component.
    """
    label: InCompArg[str]
    component: OutArg[any]

    def execute(self, ctx) -> None:
        self.component.value = gr.Video(
            label=self.label.value or "Video Input"
        )

@xai_component(color='orange')
class GradioDateTime(Component):
    """Creates a Gradio DateTime Picker.

    ##### inPorts:
    - label (str): The label for the picker.
    - include_time (bool): Whether to include time selection.

    ##### outPorts:
    - component (gr.DateTime): The created DateTime component.
    """
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
    """Creates a Gradio Color Picker.

    ##### inPorts:
    - label (str): The label for the picker.

    ##### outPorts:
    - component (gr.ColorPicker): The created ColorPicker component.
    """
    label: InCompArg[str]
    component: OutArg[any]

    def execute(self, ctx) -> None:
        self.component.value = gr.ColorPicker(
            label=self.label.value or "Color Picker"
        )

@xai_component(color='orange')
class GradioAudioRecord(Component):
    """Creates a Gradio Audio Input with recording capabilities.

    ##### inPorts:
    - label (str): The label for the audio input.
    - sources (list): List of audio sources allowed (e.g., ["microphone", "upload"]).
    - type (str): Type of input ("filepath" or "numpy").

    ##### outPorts:
    - component (gr.Audio): The created Audio component with recording capabilities.
    """
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
    """Creates a Gradio Dataframe Input/Output.

    ##### inPorts:
    - headers (dynalist): List of column headers for the dataframe.
    - datatype (dynalist): Data types for each column.

    ##### outPorts:
    - component (gr.Dataframe): The created Dataframe component.
    """
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
    """Creates a JSON Output.

    ##### inPorts:
    - label (str): The label for the output.

    ##### outPorts:
    - component (gr.JSON): The created JSON component.
    """
    label: InCompArg[str]
    component: OutArg[any]

    def execute(self, ctx) -> None:
        self.component.value = gr.JSON(label=self.label.value)

@xai_component(color='orange')
class GradioPlotOutput(Component):
    """Creates a Gradio Plot Output.

    ##### inPorts:
    - label (str): The label for the plot.

    ##### outPorts:
    - component (gr.Plot): The created Plot component.
    """
    label: InCompArg[str]
    component: OutArg[any]

    def execute(self, ctx) -> None:
        self.component.value = gr.Plot(label=self.label.value)

@xai_component(color='orange')
class GradioGallery(Component):
    """Creates a Gradio Gallery Input/Output.

    ##### inPorts:
    - label (str): The label for the gallery.

    ##### outPorts:
    - component (gr.Gallery): The created Gallery component.
    """
    label: InCompArg[str]
    component: OutArg[any]

    def execute(self, ctx) -> None:
        self.component.value = gr.Gallery(label=self.label.value)

@xai_component(color='purple')
class GradioResponses(Component):
    """Stores predefined responses in the context.

    ##### inPorts:
    - responses_dict (dict): Dictionary of predefined responses.
    """
    responses_dict: InArg[dict]

    def execute(self, ctx) -> None:
        ctx['gradio_responses'] = self.responses_dict.value or {}
