from datasets import load_dataset
import os
import hashlib
import json
from datasketch import MinHash, MinHashLSH

# Set output directory
output_dir = "data/climblab_sample"
os.makedirs(output_dir, exist_ok=True)

try:
    # Load the ClimbLab dataset (default split)
    print("Loading ClimbLab dataset...")
    dataset = load_dataset("nvidia/ClimbLab", split="train", streaming=True)
    
    # Initialize MinHash LSH
    lsh = MinHashLSH(threshold=0.5, num_perm=128)
    
    # Track unique samples
    unique_hashes = set()
    deduplicated_samples = []
    
    print("Processing samples for deduplication...")
    sample_count = 0
    
    for sample in dataset:
        sample_count += 1
        if sample_count % 1000 == 0:
            print(f"Processed {sample_count} samples, kept {len(deduplicated_samples)}")
        
        # Check if sample has the expected text field
        # You may need to adjust this based on actual dataset structure
        if 'text' not in sample:
            print(f"Warning: Sample {sample_count} missing 'text' field. Available keys: {list(sample.keys())}")
            continue
            
        text = sample['text']
        
        # Skip empty or very short texts
        if not text or len(text.strip()) < 10:
            continue
        
        # Hashing for exact deduplication
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        if text_hash in unique_hashes:
            continue  # Skip exact duplicates
        
        unique_hashes.add(text_hash)
        
        # MinHash for near-duplicate detection
        minhash = MinHash(num_perm=128)
        words = text.lower().split()  # Normalize case and split
        
        if len(words) < 5:  # Skip very short texts for MinHash
            continue
            
        for word in words:
            minhash.update(word.encode('utf-8'))
        
        # Check for similarity with existing samples
        similar_samples = lsh.query(minhash)
        if similar_samples:
            continue  # Skip if similar sample found
        
        # Add to deduplicated samples and LSH
        deduplicated_samples.append(sample)
        lsh.insert(text_hash, minhash)
        
        # Optional: Limit number of samples for testing
        if len(deduplicated_samples) >= 10000:  # Remove this line for full processing
            print("Reached sample limit for testing")
            break

    # Save the deduplicated dataset
    sample_path = os.path.join(output_dir, "climblab_deduplicated.jsonl")
    print(f"Saving {len(deduplicated_samples)} deduplicated samples...")
    
    with open(sample_path, 'w', encoding='utf-8') as f:
        for sample in deduplicated_samples:
            # Convert sample dict to JSON string
            json_line = json.dumps(sample, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"Successfully saved {len(deduplicated_samples)} deduplicated samples to {sample_path}")
    print(f"Original samples processed: {sample_count}")
    print(f"Deduplication ratio: {len(deduplicated_samples)/sample_count:.2%}")

except Exception as e:
    print(f"Error occurred: {str(e)}")
    print("This might happen if:")
    print("1. The dataset doesn't exist or is not accessible")
    print("2. The dataset structure is different than expected")
    print("3. Network connection issues")
    print("4. Insufficient permissions for file writing")