# Stable Diffusion in Diffusers library
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler

model_id_inpaint = "stabilityai/stable-diffusion-2-inpainting"

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(model_id_inpaint)
scheduler_inpaint = DPMSolverMultistepScheduler.from_config(pipe_inpaint.scheduler.config)


import gc

text_encoder_inpaint = pipe_inpaint.text_encoder
text_encoder_inpaint.eval()
unet_inpaint = pipe_inpaint.unet
unet_inpaint.eval()
vae_inpaint = pipe_inpaint.vae
vae_inpaint.eval()

del pipe_inpaint
gc.collect();

# Convert models to OpenVINO Intermediate representation (IR) format

from pathlib import Path
import torch
import numpy as np
import openvino as ov

sd2_inpainting_model_dir = Path("sd2_inpainting")
sd2_inpainting_model_dir.mkdir(exist_ok=True)

def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def convert_encoder(text_encoder: torch.nn.Module, ir_path: Path):
    """
    Convert Text Encoder model to IR.
    Function accepts pipeline, prepares example inputs for conversion
    Parameters:
        text_encoder (torch.nn.Module): text encoder PyTorch model
        ir_path (Path): File for storing model
    Returns:
        None
    """
    if not ir_path.exists():
        input_ids = torch.ones((1, 77), dtype=torch.long)
        # switch model to inference mode
        text_encoder.eval()

        # disable gradients calculation for reducing memory consumption
        with torch.no_grad():
            # export model
            ov_model = ov.convert_model(
                text_encoder,  # model instance
                example_input=input_ids,  # example inputs for model tracing
                input=([1, 77],),  # input shape for conversion
            )
            ov.save_model(ov_model, ir_path)
            del ov_model
            cleanup_torchscript_cache()
        print("Text Encoder successfully converted to IR")


def convert_unet(
    unet: torch.nn.Module,
    ir_path: Path,
    num_channels: int = 4,
    width: int = 64,
    height: int = 64,
):
    """
    Convert Unet model to IR format.
    Function accepts pipeline, prepares example inputs for conversion
    Parameters:
        unet (torch.nn.Module): UNet PyTorch model
        ir_path (Path): File for storing model
        num_channels (int, optional, 4): number of input channels
        width (int, optional, 64): input width
        height (int, optional, 64): input height
    Returns:
        None
    """
    dtype_mapping = {torch.float32: ov.Type.f32, torch.float64: ov.Type.f64}
    if not ir_path.exists():
        # prepare inputs
        encoder_hidden_state = torch.ones((2, 77, 1024))
        latents_shape = (2, num_channels, width, height)
        latents = torch.randn(latents_shape)
        t = torch.from_numpy(np.array(1, dtype=np.float32))
        unet.eval()
        dummy_inputs = (latents, t, encoder_hidden_state)
        input_info = []
        for input_tensor in dummy_inputs:
            shape = ov.PartialShape(tuple(input_tensor.shape))
            element_type = dtype_mapping[input_tensor.dtype]
            input_info.append((shape, element_type))

        with torch.no_grad():
            ov_model = ov.convert_model(unet, example_input=dummy_inputs, input=input_info)
        ov.save_model(ov_model, ir_path)
        del ov_model
        cleanup_torchscript_cache()
        print("U-Net successfully converted to IR")


def convert_vae_encoder(vae: torch.nn.Module, ir_path: Path, width: int = 512, height: int = 512):
    """
    Convert VAE model to IR format.
    VAE model, creates wrapper class for export only necessary for inference part,
    prepares example inputs for onversion
    Parameters:
        vae (torch.nn.Module): VAE PyTorch model
        ir_path (Path): File for storing model
        width (int, optional, 512): input width
        height (int, optional, 512): input height
    Returns:
        None
    """

    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, image):
            return self.vae.encode(x=image)["latent_dist"].sample()

    if not ir_path.exists():
        vae_encoder = VAEEncoderWrapper(vae)
        vae_encoder.eval()
        image = torch.zeros((1, 3, width, height))
        with torch.no_grad():
            ov_model = ov.convert_model(vae_encoder, example_input=image, input=([1, 3, width, height],))
        ov.save_model(ov_model, ir_path)
        del ov_model
        cleanup_torchscript_cache()
        print("VAE encoder successfully converted to IR")


def convert_vae_decoder(vae: torch.nn.Module, ir_path: Path, width: int = 64, height: int = 64):
    """
    Convert VAE decoder model to IR format.
    Function accepts VAE model, creates wrapper class for export only necessary for inference part,
    prepares example inputs for conversion
    Parameters:
        vae (torch.nn.Module): VAE model
        ir_path (Path): File for storing model
        width (int, optional, 64): input width
        height (int, optional, 64): input height
    Returns:
        None
    """

    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            return self.vae.decode(latents)

    if not ir_path.exists():
        vae_decoder = VAEDecoderWrapper(vae)
        latents = torch.zeros((1, 4, width, height))

        vae_decoder.eval()
        with torch.no_grad():
            ov_model = ov.convert_model(vae_decoder, example_input=latents, input=([1, 4, width, height],))
        ov.save_model(ov_model, ir_path)
        del ov_model
        cleanup_torchscript_cache()
        print("VAE decoder successfully converted to IR")

TEXT_ENCODER_OV_PATH_INPAINT = sd2_inpainting_model_dir / "text_encoder.xml"

if not TEXT_ENCODER_OV_PATH_INPAINT.exists():
    convert_encoder(text_encoder_inpaint, TEXT_ENCODER_OV_PATH_INPAINT)
else:
    print(f"Text encoder will be loaded from {TEXT_ENCODER_OV_PATH_INPAINT}")

del text_encoder_inpaint
gc.collect();

UNET_OV_PATH_INPAINT = sd2_inpainting_model_dir / "unet.xml"
if not UNET_OV_PATH_INPAINT.exists():
    convert_unet(unet_inpaint, UNET_OV_PATH_INPAINT, num_channels=9, width=64, height=64)
    del unet_inpaint
    gc.collect()
else:
    del unet_inpaint
    print(f"U-Net will be loaded from {UNET_OV_PATH_INPAINT}")
gc.collect();

VAE_ENCODER_OV_PATH_INPAINT = sd2_inpainting_model_dir / "vae_encoder.xml"

if not VAE_ENCODER_OV_PATH_INPAINT.exists():
    convert_vae_encoder(vae_inpaint, VAE_ENCODER_OV_PATH_INPAINT, 512, 512)
else:
    print(f"VAE encoder will be loaded from {VAE_ENCODER_OV_PATH_INPAINT}")

VAE_DECODER_OV_PATH_INPAINT = sd2_inpainting_model_dir / "vae_decoder.xml"
if not VAE_DECODER_OV_PATH_INPAINT.exists():
    convert_vae_decoder(vae_inpaint, VAE_DECODER_OV_PATH_INPAINT, 64, 64)
else:
    print(f"VAE decoder will be loaded from {VAE_DECODER_OV_PATH_INPAINT}")

del vae_inpaint
gc.collect();

# Prepare Inference Pipeline
import inspect
from typing import List, Optional, Union, Dict

import PIL
import cv2

from transformers import CLIPTokenizer
from diffusers import DiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler


def prepare_mask_and_masked_image(image: PIL.Image.Image, mask: PIL.Image.Image):
    """
    Prepares a pair (image, mask) to be consumed by the Stable Diffusion pipeline. This means that those inputs will be
    converted to ``np.array`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
    ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``np.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``np.float32`` too.

    Args:
        image (Union[np.array, PIL.Image]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array``
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array``.

    Returns:
        tuple[np.array]: The pair (mask, masked_image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """
    if isinstance(image, (PIL.Image.Image, np.ndarray)):
        image = [image]

    if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
        image = [np.array(i.convert("RGB"))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
    elif isinstance(image, list) and isinstance(image[0], np.ndarray):
        image = np.concatenate([i[None, :] for i in image], axis=0)

    image = image.transpose(0, 3, 1, 2)
    image = image.astype(np.float32) / 127.5 - 1.0

    # preprocess mask
    if isinstance(mask, (PIL.Image.Image, np.ndarray)):
        mask = [mask]

    if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
        mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
        mask = mask.astype(np.float32) / 255.0
    elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
        mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    masked_image = image * (mask < 0.5)

    return mask, masked_image

class OVStableDiffusionInpaintingPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae_decoder: ov.Model,
        text_encoder: ov.Model,
        tokenizer: CLIPTokenizer,
        unet: ov.Model,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        vae_encoder: ov.Model = None,
    ):
        """
        Pipeline for text-to-image generation using Stable Diffusion.
        Parameters:
            vae_decoder (Model):
                Variational Auto-Encoder (VAE) Model to decode images to and from latent representations.
            text_encoder (Model):
                Frozen text-encoder. Stable Diffusion uses the text portion of
                [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
                the clip-vit-large-patch14(https://huggingface.co/openai/clip-vit-large-patch14) variant.
            tokenizer (CLIPTokenizer):
                Tokenizer of class CLIPTokenizer(https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
            unet (Model): Conditional U-Net architecture to denoise the encoded image latents.
            vae_encoder (Model):
                Variational Auto-Encoder (VAE) Model to encode images to latent representation.
            scheduler (SchedulerMixin):
                A scheduler to be used in combination with unet to denoise the encoded image latents. Can be one of
                DDIMScheduler, LMSDiscreteScheduler, or PNDMScheduler.
        """
        super().__init__()
        self.scheduler = scheduler
        self.vae_decoder = vae_decoder
        self.vae_encoder = vae_encoder
        self.text_encoder = text_encoder
        self.unet = unet
        self._text_encoder_output = text_encoder.output(0)
        self._unet_output = unet.output(0)
        self._vae_d_output = vae_decoder.output(0)
        self._vae_e_output = vae_encoder.output(0) if vae_encoder is not None else None
        self.height = self.unet.input(0).shape[2] * 8
        self.width = self.unet.input(0).shape[3] * 8
        self.tokenizer = tokenizer
        self.register_to_config(_progress_bar_config={})

    def prepare_mask_latents(
        self,
        mask,
        masked_image,
        height=512,
        width=512,
        do_classifier_free_guidance=True,
    ):
        """
        Prepare mask as Unet nput and encode input masked image to latent space using vae encoder

        Parameters:
          mask (np.array): input mask array
          masked_image (np.array): masked input image tensor
          heigh (int, *optional*, 512): generated image height
          width (int, *optional*, 512): generated image width
          do_classifier_free_guidance (bool, *optional*, True): whether to use classifier free guidance or not
        Returns:
          mask (np.array): resized mask tensor
          masked_image_latents (np.array): masked image encoded into latent space using VAE
        """
        mask = torch.nn.functional.interpolate(torch.from_numpy(mask), size=(height // 8, width // 8))
        mask = mask.numpy()

        # encode the mask image into latents space so we can concatenate it to the latents
        latents = self.vae_encoder(masked_image)[self._vae_e_output]
        masked_image_latents = latents * 0.18215

        mask = np.concatenate([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = np.concatenate([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        return mask, masked_image_latents

    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: PIL.Image.Image,
        mask_image: PIL.Image.Image,
        negative_prompt: Union[str, List[str]] = None,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0,
        output_type: Optional[str] = "pil",
        seed: Optional[int] = None,
    ):
        """
        Function invoked when calling the pipeline for generation.
        Parameters:
            prompt (str or List[str]):
                The prompt or prompts to guide the image generation.
            image (PIL.Image.Image):
                 Source image for inpainting.
            mask_image (PIL.Image.Image):
                 Mask area for inpainting
            negative_prompt (str or List[str]):
                The negative prompt or prompts to guide the image generation.
            num_inference_steps (int, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (float, *optional*, defaults to 7.5):
                Guidance scale as defined in Classifier-Free Diffusion Guidance(https://arxiv.org/abs/2207.12598).
                guidance_scale is defined as `w` of equation 2.
                Higher guidance scale encourages to generate images that are closely linked to the text prompt,
                usually at the expense of lower image quality.
            eta (float, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [DDIMScheduler], will be ignored for others.
            output_type (`str`, *optional*, defaults to "pil"):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): PIL.Image.Image or np.array.
            seed (int, *optional*, None):
                Seed for random generator state initialization.
        Returns:
            Dictionary with keys:
                sample - the last generated image PIL.Image.Image or np.array
        """
        if seed is not None:
            np.random.seed(seed)
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get prompt text embeddings
        text_embeddings = self._encode_prompt(
            prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )
        # prepare mask
        mask, masked_image = prepare_mask_and_masked_image(image, mask_image)
        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, 1)
        latent_timestep = timesteps[:1]

        # get the initial random noise unless the user supplied it
        latents, meta = self.prepare_latents(latent_timestep)
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for t in self.progress_bar(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = np.concatenate([latent_model_input, mask, masked_image_latents], axis=1)
            # predict the noise residual
            noise_pred = self.unet([latent_model_input, np.array(t, dtype=np.float32), text_embeddings])[self._unet_output]
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                torch.from_numpy(noise_pred),
                t,
                torch.from_numpy(latents),
                **extra_step_kwargs,
            )["prev_sample"].numpy()
        # scale and decode the image latents with vae
        image = self.vae_decoder(latents * (1 / 0.18215))[self._vae_d_output]

        image = self.postprocess_image(image, meta, output_type)
        return {"sample": image}

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Union[str, List[str]] = None,
    ):
        """
        Encodes the prompt into text encoder hidden states.

        Parameters:
            prompt (str or list(str)): prompt to be encoded
            num_images_per_prompt (int): number of images that should be generated per prompt
            do_classifier_free_guidance (bool): whether to use classifier free guidance or not
            negative_prompt (str or list(str)): negative prompt to be encoded
        Returns:
            text_embeddings (np.ndarray): text encoder hidden states
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # tokenize input prompts
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids

        text_embeddings = self.text_encoder(text_input_ids)[self._text_encoder_output]

        # duplicate text embeddings for each generation per prompt
        if num_images_per_prompt != 1:
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = np.tile(text_embeddings, (1, num_images_per_prompt, 1))
            text_embeddings = np.reshape(text_embeddings, (bs_embed * num_images_per_prompt, seq_len, -1))

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            max_length = text_input_ids.shape[-1]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )

            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[self._text_encoder_output]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = np.tile(uncond_embeddings, (1, num_images_per_prompt, 1))
            uncond_embeddings = np.reshape(uncond_embeddings, (batch_size * num_images_per_prompt, seq_len, -1))

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        return text_embeddings

    def prepare_latents(self, latent_timestep: torch.Tensor = None):
        """
        Function for getting initial latents for starting generation

        Parameters:
            latent_timestep (torch.Tensor, *optional*, None):
                Predicted by scheduler initial step for image generation, required for latent image mixing with nosie
        Returns:
            latents (np.ndarray):
                Image encoded in latent space
        """
        latents_shape = (1, 4, self.height // 8, self.width // 8)
        noise = np.random.randn(*latents_shape).astype(np.float32)
        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            noise = noise * self.scheduler.sigmas[0].numpy()
        return noise, {}

    def postprocess_image(self, image: np.ndarray, meta: Dict, output_type: str = "pil"):
        """
        Postprocessing for decoded image. Takes generated image decoded by VAE decoder, unpad it to initila image size (if required),
        normalize and convert to [0, 255] pixels range. Optionally, convertes it from np.ndarray to PIL.Image format

        Parameters:
            image (np.ndarray):
                Generated image
            meta (Dict):
                Metadata obtained on latents preparing step, can be empty
            output_type (str, *optional*, pil):
                Output format for result, can be pil or numpy
        Returns:
            image (List of np.ndarray or PIL.Image.Image):
                Postprocessed images
        """
        if "padding" in meta:
            pad = meta["padding"]
            (_, end_h), (_, end_w) = pad[1:3]
            h, w = image.shape[2:]
            unpad_h = h - end_h
            unpad_w = w - end_w
            image = image[:, :, :unpad_h, :unpad_w]
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = np.transpose(image, (0, 2, 3, 1))
        # 9. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            if "src_height" in meta:
                orig_height, orig_width = meta["src_height"], meta["src_width"]
                image = [img.resize((orig_width, orig_height), PIL.Image.Resampling.LANCZOS) for img in image]
        else:
            if "src_height" in meta:
                orig_height, orig_width = meta["src_height"], meta["src_width"]
                image = [cv2.resize(img, (orig_width, orig_width)) for img in image]
        return image

    def get_timesteps(self, num_inference_steps: int, strength: float):
        """
        Helper function for getting scheduler timesteps for generation
        In case of image-to-image generation, it updates number of steps according to strength

        Parameters:
           num_inference_steps (int):
              number of inference steps for generation
           strength (float):
               value between 0.0 and 1.0, that controls the amount of noise that is added to the input image.
               Values that approach 1.0 allow for lots of variations but will also produce images that are not semantically consistent with the input.
        """
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start
    

# Generation

from tqdm import trange


def generate_image(
    pipe: OVStableDiffusionInpaintingPipeline,
    prompt: Union[str, List[str]],
    negative_prompt: Union[str, List[str]],
    input_image: PIL.Image.Image,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 20,
    mask_width: int = 20,  # Changed mask_width to 20
    seed: int = 9999,
):
    """
    Outpaint a single image with input image, with mask only on the right side

    Parameters:
        pipe (OVStableDiffusionInpaintingPipeline): inpainting pipeline.
        prompt (str or List[str]): The prompt or prompts to guide the image generation.
        negative_prompt (str or List[str]): The negative prompt or prompts to guide the image generation.
        input_image (PIL.Image.Image): The input image to be outpainted.
        guidance_scale (float, *optional*, defaults to 7.5):
            Guidance scale as defined in Classifier-Free Diffusion Guidance(https://arxiv.org/abs/2207.12598).
            guidance_scale is defined as `w` of equation 2.
            Higher guidance scale encourages to generate images that are closely linked to the text prompt,
            usually at the expense of lower image quality.
        num_inference_steps (int, *optional*, defaults to 50): The number of denoising steps for each frame. More denoising steps usually lead to a higher quality image at the expense of slower inference.
        mask_width (int, *optional*, 21): size of border mask for inpainting on each step (modified to 21).
        seed (int, *optional*, None): Seed for random generator state initialization.
    Returns:
        PIL.Image.Image: The outpainted image.
    """

    height = input_image.height
    width = input_image.width

    current_image = input_image

    # Create mask only on the right side
    mask_image = PIL.Image.new("RGB", size=(width, height), color=(255,255,255))
    mask_image.paste(PIL.Image.new("RGB", size=(width - mask_width, height)), (0, 0))
    mask_image = mask_image.convert("RGBA")

    pipe.set_progress_bar_config(desc="Generating outpainted image...")
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=current_image,
        guidance_scale=guidance_scale,
        mask_image=mask_image,
        seed=seed,
        num_inference_steps=num_inference_steps,
    )["sample"]
    outpainted_image = images[0]
    pipe.set_progress_bar_config()

    # No need for image grid or video generation/saving as we only need one outpainted image

    return outpainted_image

def append_right_side(orig_image, output_image):
    """
    Appends the right 256 pixels of output_image to the resized_image.

    Args:
        orig_image: The image to which the right side of output_image will be appended.
        output_image: The image from which the right 256 pixels will be extracted.

    Returns:
        The combined image with the appended right side.
    """
    
    # Check if the images have the same height
    if orig_image.height != output_image.height:
        raise ValueError("Images must have the same height.")

    # Get the width of the original image
    orig_width = orig_image.width

    # Extract the right 256 pixels of output_image
    right_side = output_image.crop((output_image.width - 256, 0, output_image.width, output_image.height))

    # Create a new image with the combined width
    combined_image = PIL.Image.new(orig_image.mode, (orig_width + 256, orig_image.height))

    # Paste the original image to the left side of the combined image
    combined_image.paste(orig_image, (0, 0))

    # Paste the right side of the output image to the right side of the combined image
    combined_image.paste(right_side, (orig_width, 0))

    return combined_image
def outpaint_256_right(outpaint_input):
    last_512_pixels = outpaint_input.crop((outpaint_input.width - 512, 0, outpaint_input.width, outpaint_input.height))
    
    last_256_pixels = last_512_pixels.crop((last_512_pixels.width - 256, 0, last_512_pixels.width, last_512_pixels.height))
    black_part = PIL.Image.new("RGB", (256, last_512_pixels.height), (0, 0, 0))
  
    modified_input = PIL.Image.new("RGB", (last_512_pixels.width, last_512_pixels.height))
    modified_input.paste(last_256_pixels, (0, 0))
    
    modified_input.paste(black_part, (last_512_pixels.width - 256, 0))
    
    outpainted_image = generate_image(
        pipe=ov_pipe_inpaint,
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=modified_input,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        mask_width=mask_width,
        seed=seed,
    )
    
    output = append_right_side(outpaint_input, outpainted_image)
    return output
def outpaint(outpaint_input, repetitions=1):
  """
  Performs outpainting on the input image by calling outpaint_256_right

  Args:
      outpaint_input: The input image to be outpainted.

  Returns:
      The outpainted image.
  """

  outpainted_image = outpaint_input

  for _ in range(repetitions*2): 
    outpainted_image = outpaint_256_right(outpainted_image)

  return outpainted_image
######## START ##########

core = ov.Core()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

import ipywidgets as widgets

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value="AUTO",
    description="Device:",
    disabled=False,
)

device

ov_config = {"INFERENCE_PRECISION_HINT": "f32"} if device.value != "CPU" else {}


text_enc_inpaint = core.compile_model(TEXT_ENCODER_OV_PATH_INPAINT, device.value)
unet_model_inpaint = core.compile_model(UNET_OV_PATH_INPAINT, device.value)
vae_decoder_inpaint = core.compile_model(VAE_DECODER_OV_PATH_INPAINT, device.value, ov_config)
vae_encoder_inpaint = core.compile_model(VAE_ENCODER_OV_PATH_INPAINT, device.value, ov_config)

ov_pipe_inpaint = OVStableDiffusionInpaintingPipeline(
    tokenizer=tokenizer,
    text_encoder=text_enc_inpaint,
    unet=unet_model_inpaint,
    vae_encoder=vae_encoder_inpaint,
    vae_decoder=vae_decoder_inpaint,
    scheduler=scheduler_inpaint,
)


# Input image (replace with your actual image path)
input_image = PIL.Image.open("pic1.jpg")
resized_image = input_image.resize((512, 512))



# Prompt describing the desired content for the outpainted area
prompt = "room"
# Negative prompt (optional, to avoid unwanted elements)
negative_prompt = "blurry, low-quality, unrealistic, people:1.5, faces:1.5, hands: 1.5, fingers: 1.5, easynegative, text: 1.5, symbols: 1.5"
# Other parameters (adjust as needed)
guidance_scale = 9
num_inference_steps = 15
mask_width = 256  # pixels for mask on the right side
seed = 315  # Random seed for reproducibility (optional)




# Display or save the outpainted image
out = outpaint(resized_image,4)

out.show()
out.save("outpainted_image.jpg")  # Save the output image
