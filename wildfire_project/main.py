from src.train import train
from src.evaluate import evaluate

def main():
    print("[MAIN] Starting wildfire classification pipeline...\n")
    model = train()
    print("\n[MAIN] Training complete. Starting evaluation...\n")
    evaluate(model)
    print("\n[MAIN] Pipeline finished.")

if __name__ == "__main__":
    main()