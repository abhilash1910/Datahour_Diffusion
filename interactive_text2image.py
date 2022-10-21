from tensorflow import keras
from diffusion_tf.stable_diffusion import Text2Image
import argparse
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="Image height, in pixels",
)

parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="Image width, in pixels",
)

parser.add_argument(
    "--mp",
    default=False,
    action="store_true",
    help="Enable mixed precision (fp16 computation)",
)

parser.add_argument(
    "--jit",
    default=False,
    action="store_true",
    help="Enable XLA compilation",
)

parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="Unconditional guidance scale",
)

parser.add_argument("--steps", type=int, default=50, help="Number of diffusion steps")

parser.add_argument(
    "--seed",
    type=int,
    help="Optionally specify a seed integer for reproducible results",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="How many images to generate",
)

args = parser.parse_args()

if args.mp:
    print("Using mixed precision.")
    keras.mixed_precision.set_global_policy("mixed_float16")

generator = Text2Image(img_height=args.H, img_width=args.W, jit_compile=args.jit)

while True:
    prompt = input("Enter prompt (or enter 'exit' to exit):")
    if prompt == "exit":
        break
    fname = input("Enter file name (where to save the results):")

    print(
        f"Generating {args.batch_size} image{'' if args.batch_size == 1 else 's'} for prompt '{prompt}'"
    )
    img = generator.generate(
        prompt,
        num_steps=args.steps,
        unconditional_guidance_scale=args.scale,
        temperature=1,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    if fname.endswith(".png"):
        fname = fname[:-4]
    if args.batch_size == 1:
        Image.fromarray(img[0]).save(f"{fname}.png")
        print(f"saved at {fname}.png")
    else:
        for i in range(args.batch_size):
            fname_i = f"{fname}_{i}.png"
            Image.fromarray(img[i]).save(fname_i)
            print(f"Saved at {fname_i}")
