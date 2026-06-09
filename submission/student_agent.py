from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Stejná architektura sítě, která byla použita při trénování.
    """

    def __init__(self, board_size: int, hidden_size: int = 128):
        super().__init__()

        input_size = board_size * board_size
        output_size = board_size * board_size + 1

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


class StudentAgent:
    def __init__(
        self,
        board_size: int,
        checkpoint_path: str | None = None
    ):
        self.board_size = board_size
        self.action_size = board_size * board_size + 1
        self.device = torch.device("cpu")

        self.q_net = QNetwork(board_size=board_size)

        if checkpoint_path is None:
            raise ValueError(
                "checkpoint_path must point to a trained model."
            )

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model file was not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        # Podpora obou běžných způsobů ukládání modelu.
        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            model_state_dict = checkpoint["model_state_dict"]
        else:
            model_state_dict = checkpoint

        self.q_net.load_state_dict(model_state_dict)
        self.q_net.to(self.device)
        self.q_net.eval()

    def encode_state(self, observation: dict):
        """
        Converts observation["board"] to the neural-network input.

        Board values:
             1 = current agent's pieces
            -1 = opponent's pieces
             0 = empty cells
        """
        board = np.asarray(
            observation["board"],
            dtype=np.float32
        )

        expected_shape = (
            self.board_size,
            self.board_size
        )

        if board.shape != expected_shape:
            raise ValueError(
                f"Expected board shape {expected_shape}, "
                f"but received {board.shape}."
            )

        state = board.reshape(-1)

        return torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

    def select_action(self, observation: dict) -> int:
        """
        Selects the legal action with the highest predicted Q-value.
        """
        legal_actions = list(observation["legal_actions"])

        if len(legal_actions) == 0:
            raise ValueError(
                "observation['legal_actions'] must not be empty."
            )

        # Pokud je jedinou možností pass, vrátíme pass.
        if legal_actions == [observation["pass_action"]]:
            return int(observation["pass_action"])

        state = self.encode_state(observation)

        with torch.no_grad():
            q_values = self.q_net(state).squeeze(0)

        legal_actions_tensor = torch.tensor(
            legal_actions,
            dtype=torch.long,
            device=self.device
        )

        legal_q_values = q_values[legal_actions_tensor]

        best_legal_index = int(
            torch.argmax(legal_q_values).item()
        )

        action = int(legal_actions[best_legal_index])

        # Poslední bezpečnostní pojistka.
        if action not in legal_actions:
            return int(legal_actions[0])

        return action