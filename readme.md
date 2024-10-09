# MinLSTM Implementation in "Were RNNs All We Needed?"

This repository contains an implementation of the MinLSTM, as described in the paper [*Were RNNs All We Needed?*](https://arxiv.org/pdf/2410.01201). 


This project is currently under development. Features may change, and further optimizations are expected. 
Your contributions and feedback are welcome!

### Modes
- **'par'**: Parallelizable mode that allows for faster training.
- **'seq'**: Sequential mode that processes inputs in a traditional way.


### Usage Example

```python
from minLSTMNet import MinLSTM
import torch

# Define parameters
input_size = 3   
hidden_size = 6 
seq_len = 100    
batch_size = 64  

# Create random input tensor
x = torch.randn(batch_size, seq_len, input_size)

# Initialize the MinLSTM model
model = MinLSTM(input_size=input_size, hidden_size=hidden_size, mode="par", batch_first=True)

# Forward pass through the model
output, _ = model(x)

# Print output shape
print("Output shape: ", output.shape)
```


## Citation

	@inproceedings{Feng2024WereRA,
	    title   = {Were RNNs All We Needed?},
	    author  = {Leo Feng and Frederick Tung and Mohamed Osama Ahmed and Yoshua Bengio and Hossein Hajimirsadegh},
	    year    = {2024},
	    url     = {https://arxiv.org/pdf/2410.01201}
	}
