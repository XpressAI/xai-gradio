from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, secret, xai_component, dynalist, SubGraphExecutor
import gradio as gr
import makefun
import datetime

@xai_component(color='blue', type='branch')
class GradioChatInterface(Component):
    """Creates a Gradio Chatbot interface with support for text and image uploads.
    
    ##### outPorts:
    - interface (gr.Blocks): Gradio Blocks object for the chatbot interface.
    - message (str): Stores the user's message.
    - history (list): Chat history for the session.
    - uploaded_image (any): Uploaded image file from the user.

    ##### Branches:
    - fn (BaseComponent): Subgraph executor for processing messages.
    """
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
class GradioLLMFnReturn(Component):
    """
    Expects an already-generated LLM response (from another component).
    - If the user message is missing or empty, raises an exception.
    - If no LLM response is available, raises an exception.
    - Otherwise, updates chat history, logs the conversation, 
      and places the final LLM reply in `ctx['__gradio__result__']`.

    ##### inPorts:
    - message (str): The user's message.
    - history (list): Chat history so far.
    - llm_response (str | dict | any): The LLM’s result (from another component).
    - log_file (str): Path to save chat history logs.

    ##### outPorts:
    - (none): The final reply is placed in `ctx['__gradio__result__']`.
    """
    message: InArg[str]
    history: InArg[list]
    llm_response: InArg[any]
    log_file: InArg[str]

    def execute(self, ctx) -> None:
        msg = self.message.value
        if not msg or not msg.strip():
            raise ValueError("LLMFnReturn: No valid user message found.")
        msg = msg.strip()

        chat_history = self.history.value or []
        raw_llm_response = self.llm_response.value
        if not raw_llm_response:
            raise ValueError("LLMFnReturn: No LLM response provided.")

        # Convert the response to a string if needed
        if isinstance(raw_llm_response, dict):
            # Example: if the LLM response is in a key "content"
            raw_llm_response = raw_llm_response.get("content", None)
            if not raw_llm_response:
                raise ValueError("LLMFnReturn: LLM response dict missing 'content' key.")

        reply = str(raw_llm_response)

        # Update chat history
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry_user = f"[{timestamp}] User: {msg}"
        new_entry_bot = f"[{timestamp}] Bot: {reply}"
        updated_history = chat_history + [new_entry_user, new_entry_bot]

        # Store the result in the context for Gradio
        ctx['__gradio__result__'] = [reply]
        ctx['history'] = updated_history

        # Optionally log the conversation
        log_file_path = self.log_file.value or "chat_history.log"
        try:
            with open(log_file_path, "a") as f:
                f.write(f"{new_entry_user}\n")
                f.write(f"{new_entry_bot}\n")
            print(f"LLM conversation logged to {log_file_path}")
        except Exception as e:
            print(f"LLMFnReturn: Failed to write log to {log_file_path}: {str(e)}")

@xai_component(color='purple')
class GradioAgentFnReturn(Component):
    """
    Integrates an agent (retrieved from context) and updates the chat history.

    - If `agent` or `agent_run` is missing from ctx, raises an exception.
    - If the message is empty, raises an exception.
    - After running the agent, updates `ctx['__gradio__result__']` with the agent's reply,
      logs to file, etc.

    ##### inPorts:
    - message (str): The user's message.
    - history (list): Chat history so far.
    - log_file (str): Optional path to save chat logs.

    Agent context is expected under `ctx['agent']`.
    The agent run object is expected under `ctx['agent_run']`.
    Typically, these are pre-configured earlier in the pipeline.
    """
    message: InArg[str]
    history: InArg[list]
    log_file: InArg[str]

    def execute(self, ctx) -> None:
        msg = self.message.value
        if not msg or not msg.strip():
            raise ValueError("AgentFnReturn: No valid user message found.")
        msg = msg.strip()

        chat_history = self.history.value or []
        agent_context = ctx.get('agent')
        agent_run = ctx.get('agent_run')
        if not agent_context or not agent_run:
            raise ValueError("AgentFnReturn: Missing agent or agent_run in the context.")

        # Append message to conversation
        conversation = ctx.get('agent_conversation', [])
        conversation.append({"role": "user", "content": msg})

        # Configure agent_run
        agent_run.conversation.value = conversation
        if 'name' in agent_context:
            agent_run.agent_name.value = agent_context['name']

        # Execute the agent
        agent_run.execute(ctx)
        # Retrieve agent's reply
        reply = agent_run.last_response.value
        if not reply:
            raise ValueError("AgentFnReturn: Agent did not produce a response.")

        # Update conversation in context
        ctx['agent_conversation'] = agent_run.out_conversation.value

        # Update the chat history
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry_user = f"[{timestamp}] User: {msg}"
        new_entry_bot = f"[{timestamp}] Bot: {reply}"
        updated_history = chat_history + [new_entry_user, new_entry_bot]

        # Store final result for Gradio
        ctx['__gradio__result__'] = [reply]
        ctx['history'] = updated_history

        # Log to file if desired
        log_file_path = self.log_file.value or "chat_history.log"
        try:
            with open(log_file_path, "a") as f:
                f.write(f"{new_entry_user}\n")
                f.write(f"{new_entry_bot}\n")
            print(f"Agent conversation logged to {log_file_path}")
        except Exception as e:
            print(f"AgentFnReturn: Failed to write log to {log_file_path}: {str(e)}")

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
