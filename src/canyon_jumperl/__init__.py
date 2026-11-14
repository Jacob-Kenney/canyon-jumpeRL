import gymnasium as gym

from .env import CanyonRacerEnv

if "CanyonRacer-v0" not in gym.registry:
    gym.register(id="CanyonRacer-v0", entry_point="canyon_jumperl.env:CanyonRacerEnv")


def main() -> None:
    print('canyon-jumperl: use gym.make("CanyonRacer-v0")')


__all__ = ["CanyonRacerEnv", "main"]
