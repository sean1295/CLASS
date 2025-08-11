import torch
import torch.nn as nn
from typing import List, Type, Optional, Union

class MLP(nn.Module):
    """
    A highly customizable Multi-Layer Perceptron (MLP) module for 2D inputs.

    This class allows for dynamic creation of an MLP with a variable number of
    hidden layers of a consistent size, customizable activation functions,
    optional batch normalization, and dropout. It is designed to work with
    2D tensors (batch, features).

    It is also compatible with Hydra configuration, allowing activation functions
    to be specified as strings (e.g., "ReLU", "GELU") in YAML files.
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_layers: int,
        out_features: int,
        activation: Union[str, Type[nn.Module]] = "ReLU",
        use_batch_norm: bool = True,
        dropout_prob: float = 0.1,
        final_activation: Optional[Union[str, Type[nn.Module]]] = None,
        bias: bool = True
    ):
        """
        Initializes the GeneralizedMLP module.

        Args:
            in_features (int): Number of input features.
            hidden_features (int): The number of features in each hidden layer.
            num_layers (int): The number of hidden layers.
            out_features (int): Number of output features.
            activation (Union[str, Type[nn.Module]], optional): The activation
                                                    function to use in hidden layers.
                                                    Can be a string name or a class.
                                                    Defaults to "ReLU".
            use_batch_norm (bool, optional): If True, adds a BatchNorm1d layer after each
                                             hidden layer's linear transformation.
                                             Defaults to True.
            dropout_prob (float, optional): The probability for the dropout layer after
                                            each hidden layer's activation.
                                            Defaults to 0.1.
            final_activation (Optional[Union[str, Type[nn.Module]]], optional):
                                                                    An optional activation
                                                                    function for the output.
                                                                    Defaults to None.
            bias (bool, optional): If True, adds a learnable bias to the linear layers.
                                   Defaults to True.
        """
        super().__init__()

        # --- Resolve activation functions from string names ---
        def _resolve_activation(act: Optional[Union[str, Type[nn.Module]]]) -> Optional[Type[nn.Module]]:
            if act is None:
                return None
            if isinstance(act, str):
                act_cls = getattr(nn, act, None)
                if act_cls is None:
                    raise ValueError(f"Unknown activation function: '{act}'")
                if not issubclass(act_cls, nn.Module):
                     raise TypeError(f"'{act}' is not a valid nn.Module subclass.")
                return act_cls
            elif issubclass(act, nn.Module):
                return act
            else:
                 raise TypeError(f"Activation must be a string or nn.Module subclass, not {type(act)}")

        activation_cls = _resolve_activation(activation)
        final_activation_cls = _resolve_activation(final_activation)

        # --- Build the network layers ---
        layers = []
        current_in_features = in_features
        
        hidden_features_list = [hidden_features] * num_layers

        # --- Create Hidden Layers ---
        for h_features in hidden_features_list:
            layers.append(nn.Linear(current_in_features, h_features, bias=bias))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_features))
            if activation_cls:
                layers.append(activation_cls())
            if dropout_prob > 0:
                layers.append(nn.Dropout(p=dropout_prob))
            current_in_features = h_features

        # --- Create Output Layer ---
        layers.append(nn.Linear(current_in_features, out_features, bias=bias))

        # --- Optional Final Activation ---
        if final_activation_cls is not None:
            layers.append(final_activation_cls())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the MLP.

        Handles both 2D and 3D inputs. For 3D inputs, it flattens the
        sequence dimension, processes the features, and then reshapes it back.

        Args:
            x (torch.Tensor): The input tensor. Can have shape (B, D) or (B, T, D),
                              where B=batch size, T=sequence length, D=features.

        Returns:
            torch.Tensor: The output tensor from the MLP.
        
        Raises:
            ValueError: If the input tensor dimension is not 2 or 3.
        """
        # Store original shape if input is 3D
        is_3d = x.ndim == 3
        if is_3d:
            B, T, D = x.shape
            # Flatten the batch and sequence dimensions to process features
            # Shape becomes (B * T, D)
            x = x.reshape(B * T, D)
        elif x.ndim != 2:
            raise ValueError(f"Input dimension {x.ndim} should be 2 or 3.")

        # Pass through the sequential MLP model
        out = self.mlp(x)

        # Reshape back to 3D if that was the original input format
        if is_3d:
            # Shape becomes (B, T, out_features)
            out = out.view(B, T, -1)

        return out