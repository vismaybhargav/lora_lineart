from inspect import cleandoc

from nodes import KSampler, VAEDecode, VAEEncode, CLIPTextEncode, ConditioningZeroOut
from comfy_extras.nodes_flux import FluxKontextImageScale, FluxGuidance
from comfy_extras.nodes_edit_model import ReferenceLatent
from comfy.samplers import KSampler as KSample
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
                "model": ("MODEL", {"tooltip": "The model to use for generating the image."}),
                "vae": ("VAE",),
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
                    KSample.SAMPLERS,
                    {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."},
                ),
                "scheduler": (
                    KSample.SCHEDULERS,
                    {"tooltip": "The scheduler controls how noise is gradually removed to form the image."},
                ),
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
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
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
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        guidance,
        clip,
        prompt,
    ):
        vae_encode = VAEEncode()
        vae_decode = VAEDecode()
        ksampler = KSampler()
        flux_kontext_image_scale = FluxKontextImageScale()
        clip_text_encode = CLIPTextEncode()
        reference_latent = ReferenceLatent()
        flux_guidance = FluxGuidance()
        conditioning_zero_out = ConditioningZeroOut()

        (cond_text,) = clip_text_encode.encode(clip, prompt)
        (cond_text_neg,) = conditioning_zero_out.zero_out(cond_text)

        frames = []
        if isinstance(images, torch.Tensor):
            # [B, H, W, C]
            if images.ndim == 4:
                for i in range(images.shape[0]):
                    frames.append(images[i])
            # [H, W, C] not a batch
            elif images.ndim == 3:
                frames.append(images)
            else:
                raise ValueError(f"Unexpected image shape! Getting shape: {tuple(images.shape)}")
        elif isinstance(images, (list, tuple)):
            for im in images:
                if not isinstance(im, torch.Tensor):
                    raise TypeError("Each image must be a torch.Tensor")
                if im.ndim == 4 and im.shape[0] == 1:
                    im = im[0]
                if im.ndim != 3:
                    raise ValueError(f"Unexpected per-frame shape: {tuple(im.shape)}")
                frames.append(im)
        else:
            raise TypeError("Images must be torch.Tensor or list/tuple of tensors.")

        decoded_frames = []
        for im in images:
            if im.ndim == 3:
                im_b = im.unsqueeze(0)
            else:
                im_b = im

            (scaled_im,) = flux_kontext_image_scale.scale(im_b)

            (encoded_latent,) = vae_encode.encode(vae, scaled_im)

            (cond_with_ref_pos,) = reference_latent.append(conditioning=cond_text, latent=encoded_latent)
            (cond_pos,) = flux_guidance.append(cond_with_ref_pos, guidance)

            (cond_with_ref_neg,) = reference_latent.append(cond_text_neg, encoded_latent)
            (cond_neg,) = flux_guidance.append(cond_with_ref_neg, guidance)

            (new_latent,) = ksampler.sample(model, seed, steps, cfg, sampler_name, scheduler, cond_pos, cond_neg, encoded_latent, denoise)

            (decoded,) = vae_decode.decode(vae, new_latent)

            if decoded.ndim == 4 and decoded.shape[0] == 1:
                decoded_frame = decoded[0]
            elif decoded.ndim == 3:
                decoded_frame = decoded
            else:
                raise ValueError(f"Unexpected decoded shape: {tuple(decoded.shape)}")

            decoded_frames.append(decoded_frame)

        out = torch.stack(decoded_frames, dim=0)
        return (out,)

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
