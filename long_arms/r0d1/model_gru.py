
import torch
import torch.nn as nn

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.collections import namedarraytuple
from rlpyt.models.conv2d import Conv2dHeadModel
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.dueling import DuelingHeadModel


RnnState = namedarraytuple("RnnState", ["h"])


class GRUModel(torch.nn.Module):
    """2D convolutional neural network (for multiple video frames per
    observation) feeding into an GRU and MLP output for Q-value outputs for
    the action set. Ability to track intermediate variables"""
    def __init__(
            self,
            image_shape,
            output_size,
            fc_size=512,  # Between conv and lstm.
            lstm_size=512,
            head_size=512,
            use_recurrence=True,
            dueling=False,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self.use_recurrence = use_recurrence
        self.dueling = dueling
        self.conv = Conv2dHeadModel(
            image_shape=image_shape,
            channels=channels or [32, 64, 64],
            kernel_sizes=kernel_sizes or [8, 4, 3],
            strides=strides or [4, 2, 1],
            paddings=paddings or [0, 1, 1],
            use_maxpool=use_maxpool,
            hidden_sizes=fc_size,  # ReLU applied here (Steven).
        )
        if self.use_recurrence:
            self.rnn = torch.nn.GRUCell(
                input_size=(self.conv.output_size + output_size + 1),
                hidden_size=lstm_size
            )
        else:
            self.rnn = nn.Sequential(
                nn.Linear(self.conv.output_size + output_size + 1, lstm_size),
                nn.ReLU()
            )

        if dueling:
            self.head = DuelingHeadModel(lstm_size, head_size, output_size)
        else:
            self.head = MlpModel(lstm_size, head_size, output_size=output_size)

        print('model initialized', self)  # NOTE for debug purposes

        # Logging gradients
        self.prev_hs_pre_grad = None
        self.prev_hs_rec_grad = None

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        img = observation.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.

        if self.use_recurrence:
            rnn_input = torch.cat([
                conv_out.view(T, B, -1),
                prev_action.view(T, B, -1),  # Assumed onehot.
                prev_reward.view(T, B, 1),
                ], dim=2)
            init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        else:
            rnn_input = torch.cat([
                conv_out.view(T, B, -1),
                prev_action.view(T, B, -1),  # Assumed onehot.
                prev_reward.view(T, B, 1),
            ], dim=2)
            init_rnn_state = None

        rnn_out, (hn,) = self.run_rnn(rnn_input, init_rnn_state)

        q = self.head(rnn_out.view(T * B, -1))

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        next_rnn_state = RnnState(h=hn)

        return q, next_rnn_state

    def run_rnn(self, rnn_input, init_rnn_state):
        """
        Helper method to run the RNN while keeping track of intermediate
        variables
        :param rnn_input: size [T, Batch, *]
        :param init_rnn_state: [N, Batch, *] or None
        :return:
        """

        # Helper method to save gradients of intermediate variables
        def save_grad(mut_dict, t_idx):
            """
            Helper method, save grad to a dictionary
            """
            def hook(grad):
                mut_dict[t_idx] = grad.unsqueeze(0)
            return hook

        #
        seq_T = rnn_input.size(0)
        hx = None if init_rnn_state is None else init_rnn_state[0][0]
        hs_pre = []  # track the hidden states for prediction

        # Clear gradient logging for the backward pass
        self.prev_hs_rec_grad = {}
        self.prev_hs_pre_grad = {}

        # Iterate over timesteps
        for t in range(seq_T):
            # One step of RNN
            if self.use_recurrence:
                hx = self.rnn(rnn_input[t], hx)
            else:
                hx = self.rnn(rnn_input[t])

            # Create separate paths for prediction vs. recurrent h_t
            h_pre = hx * 1
            h_pre.requires_grad_()
            h_rec = hx * 1
            h_rec.requires_grad_()

            # Register backward hooks
            h_pre.register_hook(save_grad(self.prev_hs_pre_grad, t))
            h_rec.register_hook(save_grad(self.prev_hs_rec_grad, t))
            # TODO: does gradients get accumulated in the intermediate
            # variables if I don't clear it? make sure grad isn't accumulated
            # in any variables in the grad? only accumulated in the optimizer?

            # Save output
            hs_pre.append(h_pre.unsqueeze(0))

            # Hidden state for next timestep
            hx = h_rec

        # Output
        rnn_out = torch.cat(hs_pre, dim=0)
        hn = hx.unsqueeze(0) if self.use_recurrence else None

        return rnn_out, (hn,)

