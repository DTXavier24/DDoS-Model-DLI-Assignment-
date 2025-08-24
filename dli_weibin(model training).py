def train_model(X_train, y_train):
    print("Training Random Forest...")
    start_time = time.time()

    model = RandomForestClassifier(
        n_estimators=500,     # more trees → higher stability
        max_depth=60,         # deeper trees for capturing patterns
        max_features="sqrt",  # good balance for splits
        min_samples_split=2,  # allow deep branching
        min_samples_leaf=1,   # fine-grained splits
        class_weight="balanced_subsample",  # handle class imbalance better
        bootstrap=True,       # classic RF bootstrapping
        random_state=42,
        n_jobs=-1             # use all CPU cores
    )
    model.fit(X_train, y_train)

    duration = time.time() - start_time
    print(f" Training complete in {duration:.2f} seconds")
    return model

# Train model
model = train_model(X_train, y_train)
