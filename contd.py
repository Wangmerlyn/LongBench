def main():
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup model path
    if args.model_path is not None:
        args.model = args.model_path
        model_map[args.model] = args.model_path
        maxlen_map[args.model] = maxlen_map["Llama-3.1-8B-Instruct"]

    # Determine output file
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_rag_{str(args.rag)}.jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_temp{args.temperature}_no_context.jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_temp{args.temperature}_cot.jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_temp{args.temperature}.jsonl")

    print(args)

    # Load dataset
    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    data_all = [
        {
            "_id": item["_id"],
            "domain": item["domain"],
            "sub_domain": item["sub_domain"],
            "difficulty": item["difficulty"],
            "length": item["length"],
            "question": item["question"],
            "choice_A": item["choice_A"],
            "choice_B": item["choice_B"],
            "choice_C": item["choice_C"],
            "choice_D": item["choice_D"],
            "answer": item["answer"],
            "context": item["context"],
        }
        for item in dataset
    ]

    # Check existing results and continue from there
    completed_ids = set()
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                completed_ids.add(json.loads(line)["_id"])

    data = [item for item in data_all if item["_id"] not in completed_ids]

    # If no more data, print and exit
    if not data:
        print("All data has already been processed.")
        return

    # Open output file in append mode
    fout = open(out_file, "a", encoding="utf-8")

    data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    processes = []
    for rank in range(args.n_proc):
        p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    fout.close()

    # Copy output file to ./results
    if os.path.exists(out_file):
        os.makedirs("./results/", exist_ok=True)
        target_path = os.path.join("./results/", os.path.basename(out_file))
        shutil.copy(out_file, target_path)
        print(f"Output file copied to {target_path}")
    else:
        print(f"Output file {out_file} does not exist.")
