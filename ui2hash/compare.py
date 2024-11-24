import faiss
import numpy as np
from scipy.sparse import csr_matrix

from .database import fetch_all_data, fetch_ui_data_by_apk


def search_similar_uis(apk_sha256: str, top_k: int):
    # 1. get all UI data that do not belongs to the given app
    all_data = fetch_all_data()
    print(f"Total: {len(all_data)}")
    vectors = []
    meta_data = []
    for row in all_data:
        id, sha256, ui_activity_name, uihash = row
        if sha256 != apk_sha256:
            vector = np.frombuffer(uihash, dtype=np.float32)
            vectors.append(vector)
            meta_data.append((id, sha256, ui_activity_name))

    # 2. build FAISS index
    vectors = np.array(vectors, dtype=np.float32)
    # vectors = vectors + 0.55
    index = faiss.IndexFlatIP(vectors.shape[1])     # cosine similarity
    # print(f"Index is trained: {index.is_trained}")
    csr_vectors = csr_matrix(vectors)               # convert sparse to dense
    csr_vectors = csr_vectors.toarray()
    csr_vectors = csr_vectors / np.linalg.norm(csr_vectors, axis=1)[:, None]
    index.add(csr_vectors.astype(np.float32))
    # print(f"Index size: {index.ntotal}")
    print(f"Search in {len(csr_vectors)} peers")

    # 3. search for similar peers
    ui_data = fetch_ui_data_by_apk(apk_sha256)
    if not ui_data:
        return []

    # 4. get results by index
    results = []
    for row in ui_data:
        ui_activity_name, query_vector = row
        # query_vector = query_vector + 0.55
        query_vector = csr_matrix(query_vector)               # convert sparse to dense
        query_vector = query_vector.toarray()
        query_vector = query_vector / np.linalg.norm(query_vector, axis=1)[:, None]

        # faiss.normalize_L2(query_vector)
        distances, indices = index.search(query_vector.reshape(1, -1).astype(np.float32), top_k)
        # print(distances[0])
        ui_results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]
            meta = meta_data[idx]
            ui_results.append({
                "query_activity_name": ui_activity_name,
                "apk_sha256": meta[1],
                "activity_name": meta[2],
                "distance": distance
            })

        results.append(ui_results)

    return results


def print_results(results, apk):
    for result in results:
        print(f"UI: {result[0]['query_activity_name']} (APK: {apk}")
        for match in result:
            print(f"  Similar UI: {match['activity_name']} "
                f"(APK: {match['apk_sha256']}, Score: {match['distance']:.4f})")