from multiscat_ml import TrainingStats

if __name__ == "__main__":
    stats = TrainingStats()
    stats.append(train_loss=0.5, val_loss=0.6, weight_decay=0.01)
    print("Training stats initialized and updated.")
