#!/usr/bin/env python3

"""Test environment task parsing"""

import sys
sys.path.insert(0, 'src')

from rookworld_rlvr.environment.chess_env import ChessEnvironment

def main():
    chess_env = ChessEnvironment()
    
    # Test position and move
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    uci_move = "e2e4"
    
    # Create expected response
    expected = chess_env.apply_move(fen, uci_move)
    print(f"Expected response: {expected}")
    
    # Test various malformed responses
    test_cases = [
        "A: some random text",  # Malformed
        "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4+",  # Missing fields
        "A: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1+e2e4++rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 2+0.0+0+0",  # Complete
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_case}")
        parsed = chess_env.parse_prediction(test_case)
        print(f"Parsed: {parsed}")
        
        if parsed:
            from rookworld_rlvr.reward.env_reward import EnvironmentRewardComputer
            reward_computer = EnvironmentRewardComputer()
            reward, breakdown = reward_computer.compute_reward(test_case, expected)
            print(f"Reward: {reward}, Breakdown: {breakdown}")

if __name__ == "__main__":
    main()