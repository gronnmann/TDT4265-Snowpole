import torch

ckpt = torch.load("output/checkpoint_best_ema.pth", map_location="cpu")
state_dict = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
converted = {"model": state_dict}
torch.save(converted, "output/checkpoint.pt")
print("Done. Sample keys:", list(state_dict.keys())[:3])