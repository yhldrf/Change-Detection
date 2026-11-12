"""
Hilbert Order
Modified from https://github.com/PrincetonLIPS/numpy-hilbert-curve

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Kaixin Xu
Please cite our work if the code is helpful to you.
"""

import torch


def _int_to_bits(values, num_bits):
    """Convert non-negative int64 values to big-endian bit vectors."""
    shifts = torch.arange(
        num_bits - 1, -1, -1, device=values.device, dtype=torch.long
    )
    return values.long().unsqueeze(-1).bitwise_right_shift(shifts).bitwise_and(1).byte()


def _bits_to_int(bits):
    """Convert big-endian bit vectors to int64 values."""
    num_bits = bits.shape[-1]
    shifts = torch.arange(
        num_bits - 1, -1, -1, device=bits.device, dtype=torch.long
    )
    return (bits.long().bitwise_left_shift(shifts)).sum(dim=-1)


def right_shift(binary, k=1, axis=-1):
    """Right shift an array of binary values.

    Parameters:
    -----------
     binary: An ndarray of binary values.

     k: The number of bits to shift. Default 1.

     axis: The axis along which to shift.  Default -1.

    Returns:
    --------
     Returns an ndarray with zero prepended and the ends truncated, along
     whatever axis was specified."""

    # If we're shifting the whole thing, just return zeros.
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)

    # Determine the padding pattern.
    # padding = [(0,0)] * len(binary.shape)
    # padding[axis] = (k,0)

    # Determine the slicing pattern to eliminate just the last one.
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    shifted = torch.nn.functional.pad(
        binary[tuple(slicing)], (k, 0), mode="constant", value=0
    )

    return shifted


def binary2gray(binary, axis=-1):
    """Convert an array of binary values into Gray codes.

    This uses the classic X ^ (X >> 1) trick to compute the Gray code.

    Parameters:
    -----------
     binary: An ndarray of binary values.

     axis: The axis along which to compute the gray code. Default=-1.

    Returns:
    --------
     Returns an ndarray of Gray codes.
    """
    shifted = right_shift(binary, axis=axis)

    # Do the X ^ (X >> 1) trick.
    gray = torch.logical_xor(binary, shifted)

    return gray


def gray2binary(gray, axis=-1):
    """Convert an array of Gray codes back into binary values.

    Parameters:
    -----------
     gray: An ndarray of gray codes.

     axis: The axis along which to perform Gray decoding. Default=-1.

    Returns:
    --------
     Returns an ndarray of binary values.
    """

    # Loop the log2(bits) number of times necessary, with shift and xor.
    shift = 2 ** (torch.Tensor([gray.shape[axis]]).log2().ceil().int() - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, right_shift(gray, shift))
        shift = torch.div(shift, 2, rounding_mode="floor")
    return gray


def encode(locs, num_dims, num_bits):
    """Decode an array of locations in a hypercube into a Hilbert integer.

    This is a vectorized-ish version of the Hilbert curve implementation by John
    Skilling as described in:

    Skilling, J. (2004, April). Programming the Hilbert curve. In AIP Conference
      Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.

    Params:
    -------
     locs - An ndarray of locations in a hypercube of num_dims dimensions, in
            which each dimension runs from 0 to 2**num_bits-1.  The shape can
            be arbitrary, as long as the last dimension of the same has size
            num_dims.

     num_dims - The dimensionality of the hypercube. Integer.

     num_bits - The number of bits for each dimension. Integer.

    Returns:
    --------
     The output is an ndarray of uint64 integers with the same shape as the
     input, excluding the last dimension, which needs to be num_dims.
    """

    # Keep around the original shape for later.
    orig_shape = locs.shape
    if orig_shape[-1] != num_dims:
        raise ValueError(
            """
      The shape of locs was surprising in that the last dimension was of size
      %d, but num_dims=%d.  These need to be equal.
      """
            % (orig_shape[-1], num_dims)
        )

    if num_dims * num_bits > 63:
        raise ValueError(
            """
      num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
      into a int64.  Are you sure you need that many points on your Hilbert
      curve?
      """
            % (num_dims, num_bits, num_dims * num_bits)
        )

    # Convert coordinates directly to bits. Dtype reinterpretation through
    # Tensor.view(dtype) is unavailable before PyTorch 2.0.
    gray = _int_to_bits(locs.reshape(-1, num_dims), num_bits)

    # Run the decoding process the other way.
    # Iterate forwards through the bits.
    for bit in range(0, num_bits):
        # Iterate forwards through the dimensions.
        for dim in range(0, num_dims):
            # Identify which ones have this bit active.
            mask = gray[:, dim, bit]

            # Where this bit is on, invert the 0 dimension for lower bits.
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )

            # Where the bit is off, exchange the lower bits with the 0 dimension.
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    # Now flatten out.
    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))

    # Convert Gray back to binary.
    hh_bin = gray2binary(gray)

    return _bits_to_int(hh_bin).reshape(orig_shape[:-1])


def decode(hilberts, num_dims, num_bits):
    """Decode an array of Hilbert integers into locations in a hypercube.

    This is a vectorized-ish version of the Hilbert curve implementation by John
    Skilling as described in:

    Skilling, J. (2004, April). Programming the Hilbert curve. In AIP Conference
      Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.

    Params:
    -------
     hilberts - An ndarray of Hilbert integers.  Must be an integer dtype and
                cannot have fewer bits than num_dims * num_bits.

     num_dims - The dimensionality of the hypercube. Integer.

     num_bits - The number of bits for each dimension. Integer.

    Returns:
    --------
     The output is an ndarray of unsigned integers with the same shape as hilberts
     but with an additional dimension of size num_dims.
    """

    if num_dims * num_bits > 64:
        raise ValueError(
            """
      num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
      into a uint64.  Are you sure you need that many points on your Hilbert
      curve?
      """
            % (num_dims, num_bits)
        )

    # Handle the case where we got handed a naked integer.
    hilberts = torch.atleast_1d(hilberts)

    # Keep around the shape for later.
    orig_shape = hilberts.shape
    # Convert Hilbert integers directly to the bits needed by Skilling's
    # procedure without relying on Tensor.view(dtype).
    hh_bits = _int_to_bits(hilberts.ravel(), num_dims * num_bits)

    # Take the sequence of bits and Gray-code it.
    gray = binary2gray(hh_bits)

    # There has got to be a better way to do this.
    # I could index them differently, but the eventual packbits likes it this way.
    gray = gray.reshape((-1, num_bits, num_dims)).swapaxes(1, 2)

    # Iterate backwards through the bits.
    for bit in range(num_bits - 1, -1, -1):
        # Iterate backwards through the dimensions.
        for dim in range(num_dims - 1, -1, -1):
            # Identify which ones have this bit active.
            mask = gray[:, dim, bit]

            # Where this bit is on, invert the 0 dimension for lower bits.
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )

            # Where the bit is off, exchange the lower bits with the 0 dimension.
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    flat_locs = _bits_to_int(gray)

    # Return them in the expected shape.
    return flat_locs.reshape((*orig_shape, num_dims))
