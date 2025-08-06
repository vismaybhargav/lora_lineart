from inspect import cleandoc
from nodes import KSampler, VAEDecode, VAELoader, VAEEncode, folder_paths
import comfy.samplers
import torch


class LoraLineart:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Return a dictionary which contains config for all input fields.
        Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
        Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
        The type can be a list for selection.

        Returns: `dict`:
            - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
            - Value input_fields (`dict`): Contains input fields config:
                * Key field_name (`string`): Name of a entry-point method's argument
                * Value field_config (`tuple`):
                    + First value is a string indicate the type of field or a list for selection.
                    + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "This is an image"}),
                "model": (folder_paths.get_filename_list_("models"), {"tooltip": "The model to use for generating the image."}),
                "lora_name": (folder_paths.get_filename_list_("loras"), {"tooltip": "The Lora to load"}),
                "lora_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the diffusion model. This value can be negative",
                    },
                ),
                "vae_name": (list(VAELoader.vae_list()), {"tooltip": "The VAE to load"}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality.",
                    },
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"tooltip": "The scheduler controls how noise is gradually removed to form the image."},
                ),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.",
                    },
                ),
                "clip_name1": (folder_paths.get_filename_list_("text_encoders"),),
                "clip_name2": (folder_paths.get_filename_list_("text_encoders"),),
                "type": (["sdxl", "sd3", "flux", "hunyuan_video", "hidream"]),
                "prompt": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("image_output_name",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "process"

    # OUTPUT_NODE = False
    # OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "Line Art"

    def process(
        self,
        images,
        model,
        vae_name,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        denoise,
        clip_name1,
        clip_name2,
        type,
        prompt,
    ):
        # TODO: Load Nunchaku Model and Lora Loader
        # model_loader = NunchakuFluxDiTLoader.load_model(model)
        # lora = []
        vae = VAELoader().load_vae(vae_name)[0]  # vae
        vae_encode = VAEEncode()
        vae_decode = VAEDecode()
        ksampler = KSampler()
        # dual_clip_loader = DualCLIPLoader()

        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = [images]  # convert to list
            elif images.ndim == 4:
                images = [img for img in images]  # split the batch
        elif not isinstance(images, (list, tuple)):
            images = [images]

        for im in images:
            # Seems like comfy needs it to be a torch tensor
            if not isinstance(im, torch.Tensor):
                im = torch.from_numpy(im)

            if im.ndim == 3:
                im = im.unsqueeze(0)

            encoded_latent = vae_encode.encode(vae, im)[0]

            ksampler.sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, encoded_latent, denoise)

            decoded = vae_decode.decode(vae, encoded_latent)[0]

            if decoded.ndim == 4 and decoded.shape[0] == 1:
                decoded = decoded[0]

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    # @classmethod
    # def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {"LoraLineart": LoraLineart}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"LoraLineart": "Lora Lineart"}
