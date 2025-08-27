"""
Mini Validation Module - Format and content validation for P: and A: tasks

Implements prioritized validation with:
- Format checking (binary)
- Classification scoring
- Regression scoring  
- Edit distance (Levenshtein) for FEN comparison
"""

import re
import logging
from typing import List, Tuple, Dict, Optional
import chess
import chess.engine

logger = logging.getLogger(__name__)

# Validation weights based on priority
P_WEIGHTS = {
    'best_move': 4.0,    # Most important - did we get the best move?
    'format': 2.0,       # Important - is the format correct?
    'candidates': 1.5,   # Useful - are the candidate moves good?
    'evaluations': 1.0   # Nice to have - are evaluations accurate?
}

A_WEIGHTS = {
    'format': 4.0,       # Most important - correct structure
    'fen_match': 3.0,    # Important - correct resulting position
    'game_state': 2.0,   # Useful - correct terminated/truncated flags
    'reward_value': 1.0  # Nice to have - correct reward
}


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Number of edits needed to transform s1 into s2
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def validate_p_format(completion: str) -> Tuple[float, Dict]:
    """
    Validate format of P: task completion.
    
    Expected format: "M: [moves] E: [evals] B: [best]"
    
    Args:
        completion: Generated completion string
        
    Returns:
        (score, details) where score is 0.0 or 1.0 and details contains what was found
    """
    details = {
        'has_moves': False,
        'has_evals': False,
        'has_best': False
    }
    
    if not completion:
        return 0.0, details
    
    # Check for M: section with moves
    moves_match = re.search(r'M:\s*([a-h][1-8][a-h][1-8]\w*(?:\s+[a-h][1-8][a-h][1-8]\w*)*)', completion)
    if moves_match:
        details['has_moves'] = True
        details['moves'] = moves_match.group(1).split()
    
    # Check for E: section with evaluations
    evals_match = re.search(r'E:\s*([-\d\.]+(?:\s+[-\d\.]+)*)', completion)
    if evals_match:
        details['has_evals'] = True
        try:
            details['evals'] = [float(x) for x in evals_match.group(1).split()]
        except:
            details['has_evals'] = False
    
    # Check for B: section with best move
    best_match = re.search(r'B:\s*([a-h][1-8][a-h][1-8]\w*)', completion)
    if best_match:
        details['has_best'] = True
        details['best_move'] = best_match.group(1)
    
    # Score is 1.0 only if all three sections are present
    score = 1.0 if all([details['has_moves'], details['has_evals'], details['has_best']]) else 0.0
    
    logger.debug(f"P: format validation - score: {score}, details: {details}")
    
    return score, details


def validate_a_format(completion: str) -> Tuple[float, Dict]:
    """
    Validate format of A: task completion.
    
    Expected format: "[new_FEN]+[reward]+[terminated]+[truncated]"
    
    Args:
        completion: Generated completion string
        
    Returns:
        (score, details) where score is 0.0 or 1.0 and details contains what was found
    """
    details = {
        'num_sections': 0,
        'has_fen': False,
        'has_reward': False,
        'has_terminated': False,
        'has_truncated': False
    }
    
    if not completion:
        return 0.0, details
    
    # Split by + delimiter
    sections = completion.split("+")
    details['num_sections'] = len(sections)
    
    if len(sections) >= 4:
        # Check FEN (should have / characters)
        if "/" in sections[0]:
            details['has_fen'] = True
            details['new_fen'] = sections[0].strip()
        
        # Check reward (should be numeric)
        try:
            details['reward'] = float(sections[1].strip())
            details['has_reward'] = True
        except:
            pass
        
        # Check terminated (should be true/false)
        terminated = sections[2].strip().lower()
        if terminated in ['true', 'false', '0', '1']:
            details['has_terminated'] = True
            details['terminated'] = terminated in ['true', '1']
        
        # Check truncated (should be true/false)
        truncated = sections[3].strip().lower()
        if truncated in ['true', 'false', '0', '1']:
            details['has_truncated'] = True
            details['truncated'] = truncated in ['true', '1']
    
    # Score is 1.0 only if all four sections are present and valid
    score = 1.0 if all([details['has_fen'], details['has_reward'], 
                        details['has_terminated'], details['has_truncated']]) else 0.0
    
    logger.debug(f"A: format validation - score: {score}, details: {details}")
    
    return score, details


def validate_p_best_move(
    fen: str,
    best_move: str,
    stockfish_path: Optional[str] = None
) -> float:
    """
    Validate best move accuracy (classification).
    
    Args:
        fen: Chess position
        best_move: Predicted best move
        stockfish_path: Path to Stockfish executable
        
    Returns:
        Score from 0.0 to 1.0 based on move quality
    """
    try:
        board = chess.Board(fen)
        
        # Check if move is legal
        try:
            move = chess.Move.from_uci(best_move)
            if move not in board.legal_moves:
                return 0.0
        except:
            return 0.0
        
        # If no Stockfish, at least give credit for legal move
        if not stockfish_path:
            return 0.2
        
        # Use Stockfish to evaluate
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            info = engine.analyse(board, chess.engine.Limit(depth=15), multipv=5)
            top_moves = [pv.move.uci() for pv in info[:5]]
            
            if best_move == top_moves[0]:
                return 1.0  # Perfect - best move
            elif best_move in top_moves[:3]:
                return 0.7  # Good - top 3
            elif best_move in top_moves:
                return 0.5  # OK - top 5
            else:
                return 0.2  # Legal but not great
                
    except Exception as e:
        logger.error(f"Error validating best move: {e}")
        return 0.0


def validate_p_candidates(
    fen: str,
    moves: List[str],
    stockfish_path: Optional[str] = None
) -> float:
    """
    Validate candidate moves (classification).
    
    Args:
        fen: Chess position
        moves: List of candidate moves
        stockfish_path: Path to Stockfish executable
        
    Returns:
        Score from 0.0 to 1.0 based on % of moves in top 5
    """
    if not moves:
        return 0.0
    
    try:
        board = chess.Board(fen)
        
        # Check which moves are legal
        legal_moves = []
        for move_uci in moves:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    legal_moves.append(move_uci)
            except:
                pass
        
        if not legal_moves:
            return 0.0
        
        # If no Stockfish, score based on legality
        if not stockfish_path:
            return len(legal_moves) / len(moves) * 0.3
        
        # Use Stockfish to get top moves
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            info = engine.analyse(board, chess.engine.Limit(depth=15), multipv=5)
            top_moves = [pv.move.uci() for pv in info[:5]]
            
            # Count how many are in top 5
            matches = sum(1 for move in legal_moves if move in top_moves)
            
            return matches / len(legal_moves)
            
    except Exception as e:
        logger.error(f"Error validating candidates: {e}")
        return 0.0


def validate_p_evaluations(
    fen: str,
    evals: List[float],
    stockfish_path: Optional[str] = None
) -> float:
    """
    Validate evaluation accuracy (regression).
    
    Args:
        fen: Chess position  
        evals: List of position evaluations
        stockfish_path: Path to Stockfish executable
        
    Returns:
        Score from 0.0 to 1.0 based on evaluation accuracy
    """
    if not evals:
        return 0.0
    
    if not stockfish_path:
        # Can't validate without engine
        return 0.1 if evals else 0.0
    
    try:
        board = chess.Board(fen)
        
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            info = engine.analyse(board, chess.engine.Limit(depth=15), multipv=len(evals))
            
            true_evals = []
            for pv in info[:len(evals)]:
                if pv.score.relative.score() is not None:
                    # Convert to centipawns
                    true_evals.append(pv.score.relative.score() / 100.0)
                else:
                    # Mate score
                    true_evals.append(10.0 if pv.score.relative.mate() > 0 else -10.0)
            
            if not true_evals:
                return 0.0
            
            # Calculate mean squared error
            errors = []
            for pred, true in zip(evals[:len(true_evals)], true_evals):
                # Normalize error by magnitude
                error = abs(pred - true) / (abs(true) + 1.0)
                errors.append(error)
            
            avg_error = sum(errors) / len(errors)
            
            # Convert to score (lower error = higher score)
            score = max(0.0, 1.0 - avg_error)
            
            return score
            
    except Exception as e:
        logger.error(f"Error validating evaluations: {e}")
        return 0.0


def validate_a_fen(expected: str, generated: str) -> float:
    """
    Validate FEN accuracy using edit distance.
    
    Args:
        expected: Expected FEN string
        generated: Generated FEN string
        
    Returns:
        Score from 0.0 to 1.0 based on similarity
    """
    if not expected or not generated:
        return 0.0
    
    if expected == generated:
        return 1.0
    
    # Calculate edit distance
    distance = levenshtein_distance(expected, generated)
    max_len = max(len(expected), len(generated))
    
    # Convert to similarity score
    similarity = 1.0 - (distance / max_len)
    
    return max(0.0, similarity)


def validate_a_flags(
    fen: str,
    move: str,
    terminated_str: str,
    truncated_str: str
) -> float:
    """
    Validate game state flags (classification).
    
    Args:
        fen: Starting position
        move: Move made
        terminated_str: Predicted terminated flag
        truncated_str: Predicted truncated flag
        
    Returns:
        Score from 0.0 to 1.0 based on correctness
    """
    try:
        board = chess.Board(fen)
        
        # Check move legality
        try:
            move_obj = chess.Move.from_uci(move)
            is_legal = move_obj in board.legal_moves
            
            if is_legal:
                board.push(move_obj)
                expected_terminated = board.is_game_over()
                expected_truncated = False
            else:
                expected_terminated = False
                expected_truncated = True
        except:
            # Invalid move format
            expected_terminated = False
            expected_truncated = True
        
        # Parse predicted flags
        pred_terminated = terminated_str.lower() in ['true', '1']
        pred_truncated = truncated_str.lower() in ['true', '1']
        
        # Score based on correctness
        correct = 0
        if pred_terminated == expected_terminated:
            correct += 1
        if pred_truncated == expected_truncated:
            correct += 1
        
        return correct / 2.0
        
    except Exception as e:
        logger.error(f"Error validating flags: {e}")
        return 0.0


def validate_a_reward(
    fen: str,
    move: str,
    reward: float
) -> float:
    """
    Validate reward value accuracy (regression).
    
    Expected rewards:
    - 1.0 for checkmate
    - 0.5 for draw/stalemate
    - 0.001 for continuing play
    - 0.0 for illegal move
    
    Args:
        fen: Starting position
        move: Move made
        reward: Predicted reward
        
    Returns:
        Score from 0.0 to 1.0 based on accuracy
    """
    try:
        board = chess.Board(fen)
        
        # Check move and determine expected reward
        try:
            move_obj = chess.Move.from_uci(move)
            if move_obj not in board.legal_moves:
                expected_reward = 0.0
            else:
                board.push(move_obj)
                if board.is_checkmate():
                    expected_reward = 1.0
                elif board.is_stalemate() or board.is_insufficient_material():
                    expected_reward = 0.5
                else:
                    expected_reward = 0.001
        except:
            expected_reward = 0.0
        
        # Calculate error
        error = abs(reward - expected_reward)
        
        # Convert to score
        if error < 0.01:
            return 1.0
        elif error < 0.1:
            return 0.7
        elif error < 0.3:
            return 0.4
        else:
            return 0.1
            
    except Exception as e:
        logger.error(f"Error validating reward: {e}")
        return 0.0


def validate_p_task(
    fen: str,
    completion: str,
    stockfish_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Complete validation of a P: task with weighted scores.
    
    Args:
        fen: Chess position
        completion: Generated completion
        stockfish_path: Path to Stockfish executable
        
    Returns:
        Dictionary with individual and weighted scores
    """
    results = {}
    
    # Format validation
    format_score, format_details = validate_p_format(completion)
    results['format'] = format_score
    results['format_weighted'] = format_score * P_WEIGHTS['format']
    
    # Content validation (only if format is valid)
    if format_score > 0:
        # Best move
        if 'best_move' in format_details:
            best_score = validate_p_best_move(fen, format_details['best_move'], stockfish_path)
            results['best_move'] = best_score
            results['best_move_weighted'] = best_score * P_WEIGHTS['best_move']
        
        # Candidates
        if 'moves' in format_details:
            candidates_score = validate_p_candidates(fen, format_details['moves'], stockfish_path)
            results['candidates'] = candidates_score
            results['candidates_weighted'] = candidates_score * P_WEIGHTS['candidates']
        
        # Evaluations
        if 'evals' in format_details:
            evals_score = validate_p_evaluations(fen, format_details['evals'], stockfish_path)
            results['evaluations'] = evals_score
            results['evaluations_weighted'] = evals_score * P_WEIGHTS['evaluations']
    
    # Total weighted score
    total_weight = sum(P_WEIGHTS.values())
    total_score = sum(v for k, v in results.items() if k.endswith('_weighted'))
    results['total_weighted'] = total_score / total_weight
    
    logger.info(f"P: task validation - total weighted score: {results['total_weighted']:.3f}")
    
    return results


def validate_a_task(
    fen: str,
    move: str,
    history: str,
    completion: str
) -> Dict[str, float]:
    """
    Complete validation of an A: task with weighted scores.
    
    Args:
        fen: Starting chess position
        move: Move to make
        history: Move history
        completion: Generated completion
        
    Returns:
        Dictionary with individual and weighted scores
    """
    results = {}
    
    # Format validation
    format_score, format_details = validate_a_format(completion)
    results['format'] = format_score
    results['format_weighted'] = format_score * A_WEIGHTS['format']
    
    # Content validation (only if format is valid)
    if format_score > 0:
        # FEN match
        if 'new_fen' in format_details:
            # Calculate expected FEN
            try:
                board = chess.Board(fen)
                move_obj = chess.Move.from_uci(move)
                if move_obj in board.legal_moves:
                    board.push(move_obj)
                    expected_fen = board.fen()
                else:
                    expected_fen = fen  # Invalid move, position unchanged
            except:
                expected_fen = fen
            
            fen_score = validate_a_fen(expected_fen, format_details['new_fen'])
            results['fen_match'] = fen_score
            results['fen_match_weighted'] = fen_score * A_WEIGHTS['fen_match']
        
        # Game state flags
        if 'terminated' in format_details and 'truncated' in format_details:
            terminated = 'true' if format_details['terminated'] else 'false'
            truncated = 'true' if format_details['truncated'] else 'false'
            flags_score = validate_a_flags(fen, move, terminated, truncated)
            results['game_state'] = flags_score
            results['game_state_weighted'] = flags_score * A_WEIGHTS['game_state']
        
        # Reward value
        if 'reward' in format_details:
            reward_score = validate_a_reward(fen, move, format_details['reward'])
            results['reward_value'] = reward_score
            results['reward_value_weighted'] = reward_score * A_WEIGHTS['reward_value']
    
    # Total weighted score
    total_weight = sum(A_WEIGHTS.values())
    total_score = sum(v for k, v in results.items() if k.endswith('_weighted'))
    results['total_weighted'] = total_score / total_weight
    
    logger.info(f"A: task validation - total weighted score: {results['total_weighted']:.3f}")
    
    return results


if __name__ == "__main__":
    # Test validation functions
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Testing P: Format Validation ===")
    test_completions = [
        "M: e2e4 d2d4 g1f3  E: 0.3 0.35 0.28  B: e2e4",  # Valid
        "M: e2e4 d2d4",  # Missing E: and B:
        "random text",  # Invalid
    ]
    
    for comp in test_completions:
        score, details = validate_p_format(comp)
        print(f"Completion: {comp[:50]}...")
        print(f"Score: {score}, Details: {details}")
        print()
    
    print("\n=== Testing A: Format Validation ===")
    test_completions = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1+0.001+false+false",  # Valid
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # Missing sections
        "invalid",  # Invalid
    ]
    
    for comp in test_completions:
        score, details = validate_a_format(comp)
        print(f"Completion: {comp[:50]}...")
        print(f"Score: {score}, Details: {details}")
        print()
    
    print("\n=== Testing Levenshtein Distance ===")
    fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    fen3 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Same as fen1
    
    print(f"Distance between identical FENs: {levenshtein_distance(fen1, fen3)}")
    print(f"Distance between different FENs: {levenshtein_distance(fen1, fen2)}")
    print(f"Similarity score: {validate_a_fen(fen1, fen2):.3f}")