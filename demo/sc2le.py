from pysc2.env import sc2_env
from pysc2.lib import actions

def main():
    # Create an SC2 environment with visualization enabled
    with sc2_env.SC2Env(
        map_name="Simple64",  # Built-in map for simplicity
        players=[sc2_env.Agent(sc2_env.Race.terran)],  # Play as Terran
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64),
            use_feature_units=True
        ),
        step_mul=8,  # Number of game steps per action
        game_steps_per_episode=0,  # Unlimited steps
        visualize=True  # Enable visualization
    ) as env:
        # Reset the environment to start a new episode
        timesteps = env.reset()

        # Main loop for the game
        while True:
            # Get the current state (timestep)
            timestep = timesteps[0]

            # Check if the game has ended
            if timestep.last():
                break

            # Perform a no-op action (does nothing)
            action = [actions.FUNCTIONS.no_op()]

            # Send the action to the environment
            timesteps = env.step(action)

if __name__ == "__main__":
    main()