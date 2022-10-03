# EMLO Session 4

Tasks:
- [x] Generate and use a Torchscript **Traced** Model
- [x] Transforms to be taken care by the traced model
- [x] Package into a docker (CPU, < 1 GB)
- [x] Use streamlit to deploy the model



## Generating a Torchscript Scripted Model
First, we initialize the transforms for our model.

```python
self.predict_transform = torch.nn.Sequential(
	transforms.RandomCrop(32, padding=[4, ]),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(self.mean, self.std)
    )
```


We include a `forward_jit` function with the `@torch.jit.export` decorator

```python
@torch.jit.export
def forward_jit(self, x: torch.Tensor):
	with torch.no_grad():
		# transform the inputs
		x = self.predict_transform(x)
		# forward pass
		logits = self(x)
	return logits
```

Finally, to save the model, we use

```python
scripted_model = model.to_torchscript(method="script")
path = f"{cfg.paths.output_dir}/model.script.pt"
torch.jit.save(scripted_model, path)
```


## Generating a Torchscript Traced Model (Bonus)
First, we initialize the transforms for our model (same as above).

```python
self.predict_transform = torch.nn.Sequential(
	transforms.RandomCrop(32, padding=[4, ]),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(self.mean, self.std)
    )
```


We include a `forward_jit` function with the `@torch.jit.export` decorator. Also, it gives an error with the forward function being defined twice, so we slightly modify the code.

```python
@torch.jit.ignore
def forward(self, x: torch.Tensor):
	return self.net(x)

# For scripting/tracing
@torch.jit.export
def forward_jit(self, x: torch.Tensor):
	with torch.no_grad():
	    # transform the inputs
	    x = self.predict_transform(x)
	    # forward pass
	    logits = self.net(x)
	return logits
```

Finally, to save the model, we use

```python
traced_model = model.to_torchscript(method="trace", example_inputs=torch.randn(1,3,32,32))
path = f"{cfg.paths.output_dir}/model.trace.pt"
torch.jit.save(traced_model, path)
```


Note that in either case, the model can be loaded via `torch.jit.load` without any changes.

## Streamlit
To run streamlit as the front-end, we have to specify the additional argument for the model path.

```bash
streamlit run app.py --server.port 8080 -- --model logs/train/runs/2022-09-30_10-41-11/model.trace.pt
```


## Testing the Front-end
You can get CIFAR images (which are 32*32 in size) from [github](https://github.com/YoongiKim/CIFAR-10-images)


## Docker
(This only uses the traced model)
The deployment for the code (along with the traced model) has been packaged into a docker.

See the `Dockerfile` contents.

This has been uploaded to the repository. You can get it via
```
docker pull ainoob/emlov2_s4
```

To run it, simply execute
```
docker run --rm -p 8080:8080 ainoob/emlov2_s4
```
