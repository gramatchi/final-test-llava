This is a snapshot of the model-inference script used for step 3 (ckpt28800
inference), copied here so the folder is complete on its own.

It won't run from this location -- it's part of the LLaVA package (imports
llava.model, llava.conversation, llava.mm_utils, needs PyTorch/transformers,
the merged weights, and a GPU). The actual pipeline calls the live copy under
LLaVA/llava/eval/, not this one. If that gets edited, re-copy it here.
