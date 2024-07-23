#https://gist.github.com/vadimkantorov/30ea6d278bc492abf6ad328c6965613a
#https://github.com/pytorch/pytorch/issues/32867
# PyTorch bit packing inspired by np.packbits / np.unpackbits. Feature request: https://github.com/pytorch/pytorch/issues/32867

import math
import torch


def tensor_dim_slice(tensor, dim, dim_slice):
	return tensor[(dim if dim >= 0 else dim + tensor.dim()) * (slice(None),) + (dim_slice,)]


# @torch.jit.script
def packshape(shape, dim: int = -1, mask: int = 0b00000001, dtype=torch.uint8, pack=True):
	dim = dim if dim >= 0 else dim + len(shape)
	bits, nibble = (
		8 if dtype is torch.uint8 else 16 if dtype is torch.int16 else 32 if dtype is torch.int32 else 64 if dtype is torch.int64 else 0), (
		1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else 0)
	# bits = torch.iinfo(dtype).bits # does not JIT compile
	assert nibble <= bits and bits % nibble == 0
	nibbles = bits // nibble
	shape = (shape[:dim] + (int(math.ceil(shape[dim] / nibbles)),) + shape[1 + dim:]) if pack else (
				shape[:dim] + (shape[dim] * nibbles,) + shape[1 + dim:])
	return shape, nibbles, nibble


# @torch.jit.script
def packbits(tensor, dim: int = -1, mask: int = 0b00000001, out=None, dtype=torch.uint8):
	dim = dim if dim >= 0 else dim + tensor.dim()
	shape, nibbles, nibble = packshape(tensor.shape, dim=dim, mask=mask, dtype=dtype, pack=True)
	out = out if out is not None else torch.empty(shape, device=tensor.device, dtype=dtype)
	assert out.shape == shape

	if tensor.shape[dim] % nibbles == 0:
		shift = torch.arange((nibbles - 1) * nibble, -1, -nibble, dtype=torch.uint8, device=tensor.device)
		shift = shift.view(nibbles, *((1,) * (tensor.dim() - dim - 1)))
		torch.sum(tensor.view(*tensor.shape[:dim], -1, nibbles, *tensor.shape[1 + dim:]) << shift, dim=1 + dim, out=out)

	else:
		for i in range(nibbles):
			shift = nibble * i
			sliced_input = tensor_dim_slice(tensor, dim, slice(i, None, nibbles))
			sliced_output = out.narrow(dim, 0, sliced_input.shape[dim])
			if shift == 0:
				sliced_output.copy_(sliced_input)
			else:
				sliced_output.bitwise_or_(sliced_input << shift)
	return out


# @torch.jit.script
def unpackbits(tensor, dim: int = -1, mask: int = 0b00000001, shape=None, out=None, dtype=torch.uint8):
	dim = dim if dim >= 0 else dim + tensor.dim()
	shape_, nibbles, nibble = packshape(tensor.shape, dim=dim, mask=mask, dtype=tensor.dtype, pack=False)
	shape = shape if shape is not None else shape_
	out = out if out is not None else torch.empty(shape, device=tensor.device, dtype=dtype)
	assert out.shape == shape

	if shape[dim] % nibbles == 0:
		shift = torch.arange((nibbles - 1) * nibble, -1, -nibble, dtype=torch.uint8, device=tensor.device)
		shift = shift.view(nibbles, *((1,) * (tensor.dim() - dim - 1)))
		return torch.bitwise_and((tensor.unsqueeze(1 + dim) >> shift).view_as(out), mask, out=out)

	else:
		for i in range(nibbles):
			shift = nibble * i
			sliced_output = tensor_dim_slice(out, dim, slice(i, None, nibbles))
			sliced_input = tensor.narrow(dim, 0, sliced_output.shape[dim])
			torch.bitwise_and(sliced_input >> shift, mask, out=sliced_output)
	return out


if __name__ == '__main__':
	for dim in [-1, 1, 2]:
		for shape in [(2, 10, 17), (2, 16, 16)]:
			for nibble in [1, 2, 4, 8]:
				mask = (1 << nibble) - 1
				for dtype in [torch.uint8, torch.int32, torch.int64]:
					nibbles = torch.iinfo(dtype).bits // nibble

					for k in range(10):
						x = torch.randint(0, 1 << nibble, shape, dtype=dtype)

						y = packbits(x, mask=mask, dim=dim, dtype=dtype)
						z = unpackbits(y, mask=mask, dim=dim, dtype=x.dtype, shape=x.shape)
						assert torch.allclose(x, z)

						if shape[dim] % nibbles != 0:
							continue

						y = packbits(x, mask=mask, dim=dim, dtype=dtype)
						z = unpackbits(y, mask=mask, dim=dim, dtype=x.dtype)
						assert torch.allclose(x, z)
