import torch

state = torch.load("siamese_trained.pth", map_location="cpu")

print("Type of loaded object:", type(state))

if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
    print("\nTop 30 keys in state_dict:\n")
    keys = list(state.keys())
    for k in keys[:30]:
        print(k)

elif isinstance(state, dict):
    print("\nTop-level keys in dict:\n", list(state.keys()))
    if "model_state_dict" in state:
        sd = state["model_state_dict"]
        print("\nTop 30 keys in model_state_dict:\n")
        keys = list(sd.keys())
        for k in keys[:30]:
            print(k)
else:
    print("Unexpected type. Content:", state)
    
    
    # When we first tried to evaluate the model, PyTorch gave an error saying the saved weights did not match the model architecture.
    # To understand why, we used a small script (inspect_weights.py) that prints the keys inside the saved state_dict.
    # This showed that all weights were saved under the prefix backbone., which means the model during training used a backbone module.
    # After seeing this, we updated the evaluation model to use the same structure.
    # This fixed the loading error and allowed the model to run correctly.