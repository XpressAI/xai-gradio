from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, xai_component, dynalist, SubGraphExecutor

import gradio as gr
import makefun


@xai_component(color='blue')
class GradioChatInterface(Component):
    """ Creates a Chat Interface """    
    interface: OutArg[gr.Blocks]
    fn: BaseComponent
    message: OutArg[str]
    history: OutArg[list]
    
    def execute(self, ctx) -> None:
        def fn(message, history, *args, **kwargs):                        
            self.message.value = message
            self.history.value = [x for x in history]
            SubGraphExecutor(self.fn).do(ctx)
            result = ctx['__gradio__result__']
            return result[0]
        
        self.interface.value = gr.ChatInterface(fn=fn, type='messages')


@xai_component(color='blue')
class GradioInterface(Component):
    """Create a gradio interface
    
    ##### inPorts:
    - parameterNames (list[str]): names for parameters that will be available on the parameters output dict, must have the same count as the inputs
    - inputs (list[str | Component]): list of input components
    - outputs (list[str | Component]): list of output components

    ##### outPorts:
    - parameters (dict): parameters of the 

    #### Branches:
    - fn: branch to be executed

    """
    parameterNames: InArg[dynalist]
    inputs: InArg[dynalist]
    outputs: InArg[dynalist]

    interface: OutArg[gr.Blocks]
    fn: BaseComponent
    parameters: OutArg[dict]
    
    def execute(self, ctx) -> None:
        assert len(self.parameterNames.value) == len(self.inputs.value)

        sig = "fn(%s)" % (",".join(self.parameterNames.value))
        def fn_impl(*args, **kwargs):
            self.parameters.value = kwargs
            SubGraphExecutor(self.fn).do(ctx)
            result = ctx['__gradio__result__']
            if len(result) == 1:
                return result[0]
            else:
                return result

        fn = makefun.create_function(sig, fn_impl)
        
        self.interface.value = gr.Interface(
            fn=fn,
            inputs=self.inputs.value,
            outputs=self.outputs.value
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
    """ Collect the return values for a gradio function

    ##### inPorts:
    - results (list[any]): Results for the gradio function
    """
    results: InArg[dynalist]

    def execute(self, ctx) -> None:
        ctx['__gradio__result__'] = self.results.value


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


