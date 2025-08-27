#!/usr/bin/env python3
"""
Check the format of FEN notation in the dataset
"""

from dataset import LeanRookWorldDataset

def main():
    print("Checking dataset FEN format...")
    print("=" * 60)
    
    # Load dataset
    dataset = LeanRookWorldDataset()
    dataset.load()
    
    # Get samples
    samples = dataset.get_samples(20)  # Get 20 samples
    
    p_tasks = []
    a_tasks = []
    
    for sample in samples:
        if sample.startswith("P: "):
            p_tasks.append(sample)
        elif sample.startswith("A: "):
            a_tasks.append(sample)
    
    # Analyze P: tasks
    print("\n=== P: Task FEN Format Analysis ===")
    for i, task in enumerate(p_tasks[:5], 1):
        print(f"\nP: Task {i}:")
        # Extract FEN part
        if "M:" in task:
            fen_part = task.split("M:")[0].replace("P: ", "").strip()
        else:
            fen_part = task.replace("P: ", "").strip()
        
        print(f"Full task: {task[:150]}...")
        print(f"FEN: {fen_part}")
        
        # Check for dots
        if "." in fen_part and not fen_part.endswith("."):  # Dots not part of move notation
            print("⚠️  FEN contains dots (.) for padding!")
            # Show which parts have dots
            parts = fen_part.split()
            if len(parts) > 0:
                board_part = parts[0]
                print(f"Board notation: {board_part}")
                ranks = board_part.split("/")
                for j, rank in enumerate(ranks):
                    if "." in rank:
                        print(f"  Rank {8-j} has dots: {rank}")
        else:
            print("✓ FEN uses standard notation (numbers for empty squares)")
    
    # Analyze A: tasks  
    print("\n=== A: Task FEN Format Analysis ===")
    for i, task in enumerate(a_tasks[:5], 1):
        print(f"\nA: Task {i}:")
        # Extract FEN part (it's the first component after "A: " and before first "+")
        clean_task = task[3:] if task.startswith("A: ") else task
        if "+" in clean_task:
            fen_part = clean_task.split("+")[0].strip()
        else:
            # Space-separated old format
            parts = clean_task.split()
            # FEN typically has "/" chars in first part
            fen_parts = []
            for part in parts:
                if "/" in part or len(fen_parts) < 6:  # FEN has 6 parts minimum
                    fen_parts.append(part)
                else:
                    break
            fen_part = " ".join(fen_parts)
        
        print(f"Full task: {task[:150]}...")
        print(f"FEN: {fen_part}")
        
        # Check for dots
        if "." in fen_part and not fen_part.endswith("."):
            print("⚠️  FEN contains dots (.) for padding!")
            # Show which parts have dots
            parts = fen_part.split()
            if len(parts) > 0:
                board_part = parts[0]
                print(f"Board notation: {board_part}")
                ranks = board_part.split("/")
                for j, rank in enumerate(ranks):
                    if "." in rank:
                        print(f"  Rank {8-j} has dots: {rank}")
        else:
            print("✓ FEN uses standard notation (numbers for empty squares)")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Analyzed {len(p_tasks)} P: tasks and {len(a_tasks)} A: tasks")
    
    # Check a raw sample directly from dataset
    print("\n=== Raw Dataset Sample (before preprocessing) ===")
    raw_sample = dataset.dataset['train'][0]['text']
    print(f"Raw sample: {raw_sample[:200]}...")
    if "." in raw_sample.split()[0] if raw_sample else False:
        print("⚠️  Raw dataset uses dots (.) for empty squares in FEN notation!")
    
if __name__ == "__main__":
    main()