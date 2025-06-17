from collections import defaultdict
from datasketch import MinHash, MinHashLSH #pip install datasketch
# this function is chatGPT generated, not me
def deduplicate_lsh(dataset, num_perm=128, threshold=0.8):
    """
    dataset: list of documents, each is a list of tokens (e.g., words or n-grams)
    num_perm: number of permutations (hash functions) for MinHash
    threshold: Jaccard similarity threshold for candidate pairs
    
    Returns:
        List of pairs (i, j) of document indices considered similar.
    """
    # 1. Create MinHash objects for each document
    minhashes = []
    for tokens in dataset:
        m = MinHash(num_perm=num_perm)
        for token in tokens:
            m.update(str(token).encode('utf8'))
        minhashes.append(m)
    
    # 2. Create LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    
    # 3. Insert documents into LSH
    for i, m in enumerate(minhashes):
        lsh.insert(f"doc{i}", m)
    
    # 4. Query LSH for candidate pairs
    candidates = set()
    for i, m in enumerate(minhashes):
        result = lsh.query(m)
        for r in result:
            j = int(r.replace("doc", ""))
            if i < j:
                candidates.add((i, j))
    
    return list(candidates)

def dfs(adjList, visited, node):
    visited[node] = True
    for neighbor in adjList[node]:
        if not visited[neighbor]:
            dfs(adjList, visited, neighbor)

def deduplicate(dataset, threshold=0.8, ngram_size=10):
    print("Step 1: Generating n-grams...")
    N = ngram_size
    nGrams = [[] for _ in range(len(dataset))]
    for row in range(len(dataset)):
        for i in range(len(dataset[row]) - N + 1):
            nGrams[row].append(dataset[row][i : i + N])

    print("Step 2: Preparing token sets for MinHash...")
    hashed_ngrams = [[] for _ in range(len(dataset))]
    for row in range(len(dataset)):
        for ngram in nGrams[row]:
            token = " ".join(map(str, ngram))
            hashed_ngrams[row].append(token)

    print("Step 3: Comparing candidate documents...")
    edgeList = deduplicate_lsh(hashed_ngrams, threshold = threshold)

    print("Step 4: Building adjacency list...")
    adjList = [[] for _ in range(len(dataset))]
    for pair in edgeList:
        adjList[pair[0]].append(pair[1])
        adjList[pair[1]].append(pair[0])
    
    print("Step 5: Connected Components Algorithm")
    filtered_dataset = []
    visited = [False] * len(dataset)
    for i in range(len(dataset)):
        if not visited[i]:
            dfs(adjList, visited, i)
            filtered_dataset.append(dataset[i]) # normally pick the highest quality document
    return filtered_dataset

if __name__ == "__main__":
    dataset = [
    ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],          # Doc 0
    ["brown", "fox", "jumps", "over", "the", "lazy", "dog"],        # Doc 1 (high overlap with 0)
    ["the", "slow", "white", "cat", "sleeps", "under", "the", "warm", "blanket"],     # Doc 2 (not similar)
    ["quick", "brown", "fox", "jumps", "over", "the", "dog"],                        # Doc 3 (subset of 0)
    ["the", "slow", "white", "cat", "sleeps", "under", "the", "blanket"],                           # Doc 4 (subset of 2)
    ]
    filtered_dataset = deduplicate(dataset, threshold = 0.5, ngram_size=2)
    print("Filtered Dataset:", filtered_dataset)
